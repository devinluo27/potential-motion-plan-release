import numpy as np
from os.path import join
import pdb
from diffuser.guides.policies import Policy
import diffuser.utils as utils
import torch
from diffuser.guides.policies_compose import PolicyCompose
import pybullet as p
import time, einops
from pb_diff_envs.utils.maze2d_utils import  pad_traj2d_list_v2, pad_traj2d_list_v3
from diffuser.guides.kuka_plan_utils import kuka_obs_target_list
from diffuser.guides.rm2d_colchk import RM2DCollisionChecker, DynRM2DCollisionChecker
from .kuka_plan_env import KukaEnvPlanner



class RandMaze2DEnvPlanner(KukaEnvPlanner):
    '''
    Similar to kuka env planner, but slightly modified for maze2d env
    '''

    def __init__(self) -> None:
        pass

    def update_config(self, plan_env_config):
        # no redundant keys, self.__dict__ should be empty before we update
        # print('plan_env_config._dict', plan_env_config._dict.keys())
        self.__dict__.update(plan_env_config._dict)
        self.is_dyn_env = 'Dyn' in self.env_mazelist.name


    def plan_an_env(self, env, policy, use_normed_wallLoc, plan_env_config:utils.Config):
        """
        plan multiple problems in one forward (stacked in one batch)
        Args:
            env: an env instance to plan on
            policy:
        Returns:
            vis_modelout_path_list (list of np): [ [H, Dim], ] * num_problems
            
        """
        assert not use_normed_wallLoc
        self.update_config(plan_env_config)
        # maze_idx = plan_env_config['maze_idx']


        self.return_diffusion = getattr(plan_env_config, 'return_diffusion', False)
        horizon = getattr(plan_env_config, 'horizon', policy.diffusion_model.horizon)
        self.horizon = horizon
        print(f'[plan_an_env] horizon', horizon)

        print('obs_selected_dim:', self.obs_selected_dim)
        # list of numpy, extract start and target
        obs_start_list, target_list = kuka_obs_target_list(self.problems_dict, self.maze_idx, self.prob_permaze)

        ## arr here is actually tensor
        obs_start_arr, target_arr = self.fit_obs_target_tensor(obs_start_list, target_list, self.samples_perprob)
        prob_start_idx = 0

        # [B, n_w*3]
        wall_locations = self.get_wallLoc_tensor_single(prob_start_idx, self.prob_permaze, self.samples_perprob)

        assert len(self.obs_selected_dim) == obs_start_arr.shape[-1]
        assert len(self.obs_selected_dim) == target_arr.shape[-1]
        assert obs_start_arr.ndim == 2 # (B, 2)
        assert target_arr.ndim == 2 # (B, 2)

        ## ------------------------------------
        ## --------- do the planning ----------
        ## ------------------------------------

        ## NOTE cond is for the diffusion model
        cond = {
            0: obs_start_arr, # (n_e*n_s, 7)
            horizon - 1: target_arr,
        }

        other_results = {}
        other_results['replan_suc_list'] = [] # list of bool
        other_results['no_replan_suc_list'] = [] # list of bool

        # action: first action; samples: trajectory;
        start_time = time.time()
        if type(policy) == PolicyCompose:
            ## compositional
            ## wall_locations here is a flatten 1d vector
            _, samples = policy(cond, wall_locations)

        elif type(policy) == Policy:
            ## base
            samples = policy(cond, batch_size=-1, wall_locations=wall_locations,use_normed_wallLoc=use_normed_wallLoc, return_diffusion=self.return_diffusion)

            if self.return_diffusion:
                dfu_action = samples[2] # useless
                dfu_traj = samples[3]
                samples = samples[1] # overwrite
                other_results['dfu_action'] = dfu_action
                other_results['dfu_traj'] = dfu_traj
                # assert diffusion.shape() ==  B,t,H,dim
            else:
                samples = samples[1] # (action, traj)

        else:
            raise NotImplementedError()
        
        end_time = time.time()
        other_results['batch_runtime'] = end_time - start_time
        ## ------ timing ends -------
        
        self.plan_option = 1
        if self.plan_option == 1:
            opt1_dict = self.option_1_pick_1_traj(samples, env, self.prob_permaze)
            vis_modelout_path_list = opt1_dict['modelout_path_list'] # a list
            other_results['num_colck_list'] = opt1_dict['num_colck_list']
        else:
            raise NotImplementedError

        if self.do_replan:
            self.replan_pred_trajs(opt1_dict, env, wall_locations, other_results)
            vis_modelout_path_list = opt1_dict['modelout_path_list'] # a list
        
        other_results['batch_runtime'] += np.array( opt1_dict['prob_time_list'] ).sum()

        assert vis_modelout_path_list[0].shape[-1] == len(self.obs_selected_dim)

        vis_maze_idxs_list = [self.maze_idx,] * len(vis_modelout_path_list)
        ## vis_modelout_path_list[0].shape # (1, 122, 7)
        ## 3 items
        return vis_modelout_path_list, vis_maze_idxs_list, other_results
    

    




    def option_1_pick_1_traj(self, samples, env, prob_permaze):
        '''
        We plan *samples_perprob* candidates for each problems,
        now pick the successful one, * all in numpy *
        1. only append the success traj
        2. if no success append the last one
        '''
        # 1. reshape
        # samples.observations (np): unnorm [B=n_p*n_s, horizon, dim]
        config_dim =  samples.observations.shape[-1]
        new_shape = (prob_permaze, self.samples_perprob, self.horizon, config_dim)
        ## [n_p, n_candidates, horizon, dim]
        pred_trajs = samples.observations.reshape(*new_shape).copy()


        # 2. list to hold results
        modelout_path_list = [None] * prob_permaze # list of np
        replan_candidate_list = [None] * prob_permaze # list of (list of np)
        prob_time_list = [None] * prob_permaze
        num_colck_list = [None] * prob_permaze
        suc_list = [None] * prob_permaze


        # 3. collision checker, different in Dynamic
        if self.is_dyn_env: # update in above
            checker = DynRM2DCollisionChecker(normalizer=None) # already unnormed, so None
            checker.use_collision_eps = True
        else:
            checker = RM2DCollisionChecker(normalizer=None) # already unnormed, so None
            checker.use_collision_eps = True
            checker.collision_eps = 0.08

        self.replanner.set_collision_checker(checker) # shallow copy
        # print('pred_trajs', pred_trajs.shape) # e.g., pred_trajs (prob_permaze, samples_perprob, H=48, 2)

        # 4. loop through each problem in the batch
        for i_p in range(prob_permaze):
            if self.is_dyn_env:
                ## Preprocess the wall location for dynamic Env, 
                ## because the checker and model input need different format
                ## np, h, nw*d
                wl = self.pick_wtrajs_np_h_nwd(self.problems_dict['infos/wall_locations'], self.maze_idx, self.prob_permaze)
                wl = wl[self.prob_idx] if prob_permaze == 1 else wl[i_p] # h, nw*d
                ## NOTE pad wtrajs, because robot traj could be longer
                wl = pad_traj2d_list_v2( [wl,], self.horizon )[0] # e.g. (48,2) -> (60,2)
                wl = wl.reshape(wl.shape[0], -1, self.env_mazelist.world_dim) # h, nw, dim

                wl = np.transpose(wl, (1, 0, 2) ) # nw, h, dim
                # set the wall moving trajectory and wall half extent
                checker.update_wtrajs(wtrajs=wl, hExts=self.env_mazelist.hExt_list[0]) # hExt_list: n_e, nw, 2
            
            ## Timing
            tic = time.time()
            num_collision_check = 0
            
            # 5. loop through each candidate of the problem
            for i_aj in range(self.samples_perprob):
                # check if the traj is good, (*two end* and collision)
                traj = pred_trajs[i_p][i_aj]
                assert type(traj) == np.ndarray and traj.ndim == 2

                if self.is_dyn_env:
                    is_valid, num_colck = checker.check_single_traj( traj, wl )
                else:
                    # We do not check if the two ends are start and goal, 
                    # however, these are already set in diffusion process
                    is_valid, num_colck = checker.check_single_traj( traj, env )
                    # is_valid, num_colck = checker.check_1_traj_eps( traj, env )


                num_collision_check += num_colck
                if is_valid:
                    modelout_path_list[i_p] = traj
                    # print(f'prob{i_p} traj{i_aj}: success')
                    suc_list[i_p] = True
                    break
            if not suc_list[i_p]:
                replan_candidate_list[i_p] = pred_trajs[i_p]
            
            prob_time = time.time() - tic
            prob_time_list[i_p] = prob_time if self.samples_perprob > 1 else 0
            num_colck_list[i_p] = num_collision_check
            
            ## all traj fails, just use the last one
            if modelout_path_list[i_p] is None:
                modelout_path_list[i_p] = traj
                suc_list[i_p] = False

        print('suc_list', suc_list)
        print('suc_list mean', np.array(suc_list).mean())

        opt1_dict = dict(
            modelout_path_list=modelout_path_list,
            replan_candidate_list=replan_candidate_list,
            prob_time_list=prob_time_list,
            num_colck_list=num_colck_list,
            suc_list=suc_list
        )
        return opt1_dict

    
        
    def plan_kuka_save_visfig(self, renderer, maze_prefix, vis_modelout_path_list, \
        vis_maze_idxs_list, plan_env_config):
        '''
        rm2d custom, different from kuka
        '''

        self.update_config(plan_env_config)

        # assert self.plan_option == 1, 'other option may cause bugs'
        vis_start_idx = plan_env_config.get('vis_start_idx', 2)
        # total_samples = self.prob_permaze * self.samples_perprob
        num_vis = self.num_vis_traj # * samples_perprob # 1
        vis_gap = 1

        ## ---- we have finished `prob_permaze` * pertraj episodes ------

        assert len(vis_maze_idxs_list) == self.prob_permaze and len(vis_modelout_path_list) == self.prob_permaze

        assert len(np.unique(vis_maze_idxs_list)) == 1, 'should be the same'
        assert type(vis_modelout_path_list[0]) == np.ndarray, 'should be a list of np'

        str_epoch = plan_env_config['str_epoch']
        
        ## save direct prediction
        fullpath = join(self.savepath_dir, f'{maze_prefix}{self.maze_idx}-modelpred-{str_epoch}.png')
        vis_end_idx = vis_start_idx + num_vis * vis_gap
        
        # now np (n_p, horizon, 7/14)
        vis_modelout_path_list = vis_modelout_path_list[vis_start_idx:vis_end_idx:vis_gap]
        ## also need to trunc
        vis_maze_idxs_list = vis_maze_idxs_list[vis_start_idx:vis_end_idx:vis_gap] # ?a list of str if composed env?


        if 'Dyn' in self.env_mazelist.name:
            ## dynamic maze2d

            wl = self.pick_wtrajs_np_h_nwd(self.problems_dict['infos/wall_locations'], vis_maze_idxs_list[0], self.prob_permaze)
            wl = wl[vis_start_idx:vis_end_idx:vis_gap] # e.g. (2, 48, 2)

            ## adaptively algin len with robot traj
            wl = pad_traj2d_list_v3(wl, vis_modelout_path_list)

            renderer.composite(fullpath, vis_modelout_path_list, maze_idx=vis_maze_idxs_list,
                               wtrajs_list=wl)
        else:
            # static maze2d
            renderer.composite(fullpath, vis_modelout_path_list, maze_idx=vis_maze_idxs_list)
    

    def pick_wtrajs_np_h_nwd(self, all_wtrajs, maze_idx: int, prob_permaze: int):
        '''n_e, n_p, h*nw*2 -> n_p, h, nw*2'''
        # all_wtrajs: n_env, n_p, h*n_w*2
        # return np, h, nw*2
        wl = all_wtrajs[maze_idx] # already int
        nw_d = self.env_mazelist.world_dim * self.env_mazelist.num_walls

        wl = einops.rearrange( wl, 'n_p (h nw_d) -> n_p h nw_d', h=self.horizon_wtrajs, nw_d=nw_d) # self.horizon
        wl = wl[:prob_permaze, :, :].copy()
        return wl




from diffuser.guides.kuka_replan import KukaDiffusionReplanner
class RM2DReplanner(KukaDiffusionReplanner):
    pass
    # def __init__(self) -> None:
        # pass