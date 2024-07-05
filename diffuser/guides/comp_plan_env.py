import numpy as np
from os.path import join
import pdb
from diffuser.guides.policies import Policy
import diffuser.utils as utils
import torch, time, einops
from diffuser.guides.policies_compose import PolicyCompose
import pybullet as p
from pb_diff_envs.utils.maze2d_utils import pad_traj2d_list_v2
from diffuser.guides.kuka_plan_utils import kuka_obs_target_list
from diffuser.guides.rm2d_colchk import RM2DCollisionChecker, DynRM2DCollisionChecker
from diffuser.guides.kuka_colchk import KukaCollisionChecker
from .kuka_plan_env import KukaEnvPlanner
from .rm2d_plan_env import RandMaze2DEnvPlanner


class ComposedEnvPlanner(RandMaze2DEnvPlanner):
    '''
    Compositional Motion Planner
    '''

    def __init__(self) -> None:
        pass

    def update_config(self, plan_env_config):
        # no redundant keys, self.__dict__ should be empty before we update
        # print('plan_env_config._dict', plan_env_config._dict.keys())
        self.__dict__.update(plan_env_config._dict)
        self.is_dyn_env = 'Dyn' in self.env_mazelist.name
        self.is_kuka = 'kuka' in self.env_mazelist.name.lower()

    def plan_an_env(self, env, policy, use_normed_wallLoc, plan_env_config:utils.Config):
        '''
        plan multiple problems in one forward (stacked in one batch)

        Args:
            env: an env instance
            policy: policy to generate motion plans
            plan_env_config: other planning configurations

        '''
        
        assert not use_normed_wallLoc
        self.update_config(plan_env_config)
        self.samples_permaze = self.prob_permaze
        # maze_idx = plan_env_config['maze_idx']


        self.return_diffusion = getattr(plan_env_config, 'return_diffusion', False)
        horizon = getattr(plan_env_config, 'horizon', policy.diffusion_model.horizon)
        # used in dyn pad wtrajs
        self.horizon = horizon if not self.is_dyn_env else max(self.horizon_pl_list)
        print(f'[plan_an_env] horizon', horizon)
        print('obs_selected_dim:', self.obs_selected_dim)

        # list of numpy, extract start and target
        obs_start_list, target_list = kuka_obs_target_list(self.problems_dict, self.maze_idx, self.prob_permaze)

        ## arr here is actually tensor
        obs_start_arr, target_arr = self.fit_obs_target_tensor(obs_start_list, target_list, self.samples_perprob)
        prob_start_idx = 0

        # [B, n_w*3]
        wall_locations = self.get_wallLoc_tensor_single(prob_start_idx, self.samples_permaze, self.samples_perprob)

        assert len(self.obs_selected_dim) == obs_start_arr.shape[-1]
        assert len(self.obs_selected_dim) == target_arr.shape[-1]
        assert obs_start_arr.ndim == 2 # (B, 2)
        assert target_arr.ndim == 2 # (B, 2)

        ## ------------------------------------
        ## --------- do the planning ----------
        ## ------------------------------------
        ## we use multiple horizons because the distance to the target might be large
        ## set the start and goal in cond for conditional diffusion generation
        ## create a list of cond with different horizons
        cond_mulho = []
        for h in self.horizon_pl_list:
            cond = {
                0: obs_start_arr, # (n_e*n_s, 7)
                h - 1: target_arr,
            }
            cond_mulho.append(cond)

        other_results = {}
        other_results['replan_suc_list'] = [] # list of bool
        other_results['no_replan_suc_list'] = [] # list of bool

        # action: first action; samples: trajectory; batch_size should be 1
        start_time = time.time()
        if type(policy) == PolicyCompose:
            ## wall_locations here is a flatten vector
            samples_mulho = policy.call_multi_horizon( cond_mulho, wall_locations )
        else:
            raise NotImplementedError()
        
        end_time = time.time()
        other_results['batch_runtime'] = end_time - start_time
        
        ## ------ timing ends -------
        
        self.plan_option = 1
        if self.plan_option == 1:
            ## pick one sucessful motion plan (trajectory)
            opt1_dict = self.option_1_pick_1_traj(samples_mulho, env, self.samples_permaze)
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

        ## 3 items
        return vis_modelout_path_list, vis_maze_idxs_list, other_results
    
    

    def option_1_pick_1_traj(self, samples_mulho, env, samples_permaze):
        '''
        We plan samples_perprob candidates for each problems,
        now pick the successful one, * in numpy *
        1. only append the success traj
        2. if no success append the last one
        '''
        pred_trajs = [] # a list of list [5, h, dim]
        for i_h, ho in enumerate(self.horizon_pl_list):

            samples = samples_mulho[i_h]
            # 1. ---- reshape ----
            # samples.observations (np): unnorm [B=n_p*n_s, horizon, 7]
            config_dim =  samples.observations.shape[-1]
            new_shape = (samples_permaze, self.samples_perprob, ho, config_dim)
            ## reshape would be risky, manually checked, (B, n_s, h, dim)
            tmp_tj = samples.observations.reshape(*new_shape).copy() # need copy?
            ## when dyn pad if necessary
            ho_diff = self.horizon_wtrajs - tmp_tj.shape[2] # wtrajs - r_traj
            if self.is_dyn_env and ho_diff > 0:
                pad_width = ((0, 0), (0, 0), (0, ho_diff), (0, 0))
                tmp_tj = np.pad( tmp_tj, pad_width, mode='edge', )

            pred_trajs.append( list( tmp_tj ) ) # a list of (5, h1, dim)

        pred_trajs = list( zip(*pred_trajs) ) # a list of tuple: (5, h1, dim), (5, h2, dim), (5, h3, dim)
        assert len(pred_trajs) == samples_permaze
        assert len(pred_trajs[0]) == len(self.horizon_pl_list)
        for i, xx in enumerate( pred_trajs[0] ):
            print(f'pred traj[0] {i}: {xx.shape}')


        # 2. list to hold results
        modelout_path_list = [None] * samples_permaze # list of np
        replan_candidate_list = [None] * samples_permaze # list of (list of np)

        prob_time_list = [None] * samples_permaze
        num_colck_list = [None] * samples_permaze
        suc_list = [None] * samples_permaze


        # 3. --- collision checker, different in Dynamic ----
        self.num_horizons = len(self.horizon_pl_list)
        # pdb.set_trace()
        if self.is_dyn_env: # update in above
            checker = DynRM2DCollisionChecker(normalizer=None) # already unnormed
            checker_list = [DynRM2DCollisionChecker(normalizer=None) for _ in range(self.num_horizons)] # already unnormed
            checker.use_collision_eps = True
            for cker in checker_list:
                cker.use_collision_eps = True
        else:
            if self.is_kuka:
                checker = KukaCollisionChecker(normalizer=None)
            else:
                checker = RM2DCollisionChecker(normalizer=None) # already unnormed
                checker.use_collision_eps = True
                checker.collision_eps = 0.08


        self.replanner.set_collision_checker(checker) # shallow copy
        

        for i_p in range(samples_permaze):
            if self.is_dyn_env:
                ## configurate the collision checker for dynamic env
                for i_c in range(self.num_horizons):
                    # h, nw*d
                    wl = self.pick_wtrajs_np_h_nwd(self.problems_dict['infos/wall_locations'], self.maze_idx, self.prob_permaze)
                    ## from here, add multi horizon support
                    wl = wl[self.prob_idx] if samples_permaze == 1 else wl[i_p] # h, nw*d
                    ho_tmp = max(self.horizon_wtrajs, self.horizon_pl_list[i_c] ) # r_traj might be shorter
                    wl = pad_traj2d_list_v2( [wl,], ho_tmp )[0] # e.g. (48,2) -> (60,2)

                    wl = wl.reshape(wl.shape[0], -1, self.env_mazelist.world_dim) # h, nw, dim
                    wl = np.transpose(wl, (1, 0, 2) ).copy() # nw, h, dim

                    checker_list[i_c].update_wtrajs(wtrajs=wl, hExts=self.env_mazelist.hExt_list[0]) # hExt_list: n_e, nw, 2

            ## all candidates correspond to problem i_p
            traj_tmp = pred_trajs[i_p] # a tuple: (B, h1, dim), (B, h2, dim), (B, h3, dim), ...
            traj_tmp = list( zip( *traj_tmp ) ) # 5 * [ (h1, dim), (h2, dim), (h3, dim) ] 
            traj_a_prob = [] # a list of all candidates trajectories, len: num_horizons * B
            for tup in traj_tmp:
                for tj in tup:
                    traj_a_prob.append(tj)

            # print(f'{i_p} len traj_a_prob {len(traj_a_prob)}')



            tic = time.time()
            num_collision_check = 0
            ## check all trajectories of a problem
            for i_aj in range(len(traj_a_prob)): # a list
                
                # 1. check if the traj if good, (*two end* and collision)
                traj = traj_a_prob[i_aj]

                assert type(traj) == np.ndarray and traj.ndim == 2

                if self.is_dyn_env:
                    is_valid, num_colck = checker_list[i_aj % self.num_horizons].check_single_traj( traj, wl[:, :len(traj), :] )
                else:
                    # [Caution] *two end* are not checked
                    is_valid, num_colck = checker.check_single_traj( traj, env )


                num_collision_check += num_colck
                if is_valid:
                    modelout_path_list[i_p] = traj
                    # print(f'prob{i_p} traj{i_aj}: success')
                    suc_list[i_p] = True
                    break

            if not suc_list[i_p]:
                ## no motion trajectories succeed
                # pred_trajs[i_p] is a tuple of 4, each (5, 40, 2)
                replan_candidate_list[i_p] = traj_a_prob
            
            prob_time = time.time() - tic
            prob_time_list[i_p] = prob_time if self.samples_perprob > 1 else 0
            num_colck_list[i_p] = num_collision_check
            
            ## all traj fails, use the last one
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



    def plan_an_env_sequential(self, env, policy, use_normed_wallLoc, plan_env_config:utils.Config):
        """
        plan one problems each time
        Args:
            env: an env instance
            policy: policy to generate motion plans
            plan_env_config: other planning configurations
        """
        assert not use_normed_wallLoc
        self.update_config(plan_env_config)
        self.samples_permaze = self.prob_permaze


        self.return_diffusion = getattr(plan_env_config, 'return_diffusion', False)
        horizon = getattr(plan_env_config, 'horizon', policy.diffusion_model.horizon)
        self.horizon = horizon
        print(f'[plan_an_env_sequential] horizon:', horizon)

        # print('obs_selected_dim:', self.obs_selected_dim)
        # [n_probs, 2] extract start and target
        obs_start_list, target_list = kuka_obs_target_list(self.problems_dict, self.maze_idx, self.prob_permaze)

        
        
        prob_start_idx = 0
        # [np*1, n_w*3] samples_permaze, samples_perprob
        wall_locations = self.get_wallLoc_tensor_single(prob_start_idx, self.samples_permaze, 1)

        n_probs = obs_start_list.shape[0]
        vis_modelout_path_list = []
        other_results = {}
        self.other_results = other_results
        other_results['replan_suc_list'] = [] # list of bool
        other_results['no_replan_suc_list'] = [] # list of bool
        each_prob_time = [] # list of float
        each_prob_colck = []
        each_prob_nrsl = []
        each_prob_rsl = []



        for i_p in range(n_probs):
            ## arr here is actually tensor
            # [1, 2]
            obs_start_arr, target_arr = self.fit_obs_target_tensor(obs_start_list[i_p:i_p+1], target_list[i_p:i_p+1], self.samples_perprob)
            # [1, n_w*3]
            wloc = wall_locations[ i_p : i_p+1].repeat((self.samples_perprob, 1))


            assert len(self.obs_selected_dim) == obs_start_arr.shape[-1]
            assert len(self.obs_selected_dim) == target_arr.shape[-1]
            assert obs_start_arr.ndim == 2 # (B, 2)
            assert target_arr.ndim == 2 # (B, 2)

            cond_mulho = []
            for h in self.horizon_pl_list:
                cond = {
                    0: obs_start_arr, # (n_e*n_s, 7)
                    h - 1: target_arr,
                }
                cond_mulho.append(cond)

            start_time = time.time()
            # action: first action; samples: trajectory; batch_size should be 1
            if type(policy) == PolicyCompose:
                samples_mulho = policy.call_multi_horizon( cond_mulho, wloc )
                
            elif type(policy) == Policy:
                raise RuntimeError("Please use the policy without compose.")
            else:
                raise NotImplementedError()
            end_time = time.time()
            
        
            self.plan_option = 1
            if self.plan_option == 1:
                self.prob_idx = i_p
                # create opt1_dict, cnt collision
                opt1_dict = self.option_1_pick_1_traj(samples_mulho, env, samples_permaze=1)
                assert len(opt1_dict['modelout_path_list']) == 1
                single_out = opt1_dict['modelout_path_list'][0] # np 2d?
            else:
                raise NotImplementedError()
            

            if self.do_replan:
                # will update colck in opt1_dict
                self.replan_pred_trajs(opt1_dict, env, wloc, other_results)
                single_out = opt1_dict['modelout_path_list'][0] # a list of np (h, dim)

            assert len(opt1_dict['prob_time_list']) == 1
            
            ## diffusion time + (check&replan time)
            each_prob_time.append( end_time - start_time + opt1_dict['prob_time_list'][0] )
            each_prob_colck.append( opt1_dict['num_colck_list'][0]  )
            each_prob_nrsl.append( other_results['no_replan_suc_list'] )
            each_prob_rsl.append( other_results['replan_suc_list'] )
            
            vis_modelout_path_list.append(single_out)

        # averaging of the env
        other_results['batch_runtime'] = np.array(each_prob_time).sum().item()
        
        other_results['num_colck_list'] = np.array( each_prob_colck )
        
        other_results['no_replan_suc_list'] = np.array(each_prob_nrsl)
        other_results['replan_suc_list'] = np.array(each_prob_rsl)
        # print('each_prob_time:', each_prob_time)
        # print('num_colck_list:', other_results['num_colck_list'])



        assert vis_modelout_path_list[0].shape[-1] == len(self.obs_selected_dim)

        vis_maze_idxs_list = [self.maze_idx,] * len(vis_modelout_path_list)
        

        ## 3 items
        return vis_modelout_path_list, vis_maze_idxs_list, other_results





    def plan_env_interact(self, env, policy, 
                          obs_start_arr, target_arr, wall_locations,
                          use_normed_wallLoc, plan_env_config:utils.Config):
        """
        for jupyter notebook realtime interaction planning
        """
        assert not use_normed_wallLoc
        self.update_config(plan_env_config)
        self.samples_permaze = self.prob_permaze
        # maze_idx = plan_env_config['maze_idx']


        self.return_diffusion = getattr(plan_env_config, 'return_diffusion', False)
        horizon = getattr(plan_env_config, 'horizon', policy.diffusion_model.horizon)
        # used in dyn pad wtrajs
        self.horizon = horizon if not self.is_dyn_env else max(self.horizon_pl_list)
        print(f'[plan_an_env] horizon', horizon)

        print('obs_selected_dim:', self.obs_selected_dim)
        
        assert obs_start_arr.ndim == 2 # (B, 2)
        assert target_arr.ndim == 2 # (B, 2)

        ## ------------------------------------
        ## --------- do the planning ----------
        ## ------------------------------------

        ## NOTE cond is for the diffusion model
        ## create a list of cond with different horizon
        cond_mulho = []
        for h in self.horizon_pl_list:
            cond = {
                0: obs_start_arr, # (n_e*n_s, 7)
                h - 1: target_arr,
            }
            cond_mulho.append(cond)

        # print( obs_start_arr, target_arr )
        other_results = {}
        other_results['replan_suc_list'] = [] # list of bool
        other_results['no_replan_suc_list'] = [] # list of bool

        # action: first action; samples: trajectory; batch_size should be 1
        start_time = time.time()
        if type(policy) == PolicyCompose:
            samples_mulho = policy.call_multi_horizon( cond_mulho, wall_locations )

        elif type(policy) == Policy:
            assert False
        else:
            raise NotImplementedError()
        
        end_time = time.time()
        other_results['batch_runtime'] = end_time - start_time
        ## ------ timing ends -------
        
        self.plan_option = 1
        if self.plan_option == 1:

            opt1_dict = self.option_1_pick_1_traj(samples_mulho, env, self.samples_permaze)
            # opt1_dict['replan_candidate_list'] = 
            vis_modelout_path_list = opt1_dict['modelout_path_list'] # a list
            other_results['num_colck_list'] = opt1_dict['num_colck_list']
        else:
            pdb.set_trace()

        if self.do_replan:
            self.replan_pred_trajs(opt1_dict, env, wall_locations, other_results)
            vis_modelout_path_list = opt1_dict['modelout_path_list'] # a list

        other_results['batch_runtime'] += np.array( opt1_dict['prob_time_list'] ).sum()
        
        assert vis_modelout_path_list[0].shape[-1] == len(self.obs_selected_dim)
        
        
        ## 3 items
        return vis_modelout_path_list, samples_mulho, opt1_dict
    








from diffuser.guides.kuka_replan import KukaDiffusionReplanner
class ComposedRM2DReplanner(KukaDiffusionReplanner):
    '''might not be completed'''
    pass