import numpy as np
from os.path import join
import pdb
from diffuser.guides.policies import Policy
import diffuser.utils as utils
import torch
from diffuser.guides.policies_compose import PolicyCompose
from collections import OrderedDict
from tqdm import tqdm
from diffuser.utils.eval_utils import pad_and_concatenate_2d
import pybullet as p
import time
from diffuser.guides.kuka_plan_utils import kuka_obs_target_list
from diffuser.guides.kuka_colchk import KukaCollisionChecker

class KukaEnvPlanner:
    def __init__(self) -> None:
        pass

    def update_config(self, plan_env_config):

        # print('plan_env_config._dict', plan_env_config._dict.keys())
        self.__dict__.update(plan_env_config._dict)

    def plan_an_env(self, env, policy, use_normed_wallLoc, plan_env_config:utils.Config):
        """
        plan multiple problems in one forward (stacked in one batch)
        Args:
            env: an env instance
            policy: policy to generate motion plans
            plan_env_config: other planning configurations
        
        """
        self.update_config(plan_env_config)


        self.return_diffusion = getattr(plan_env_config, 'return_diffusion', False)
        horizon = getattr(plan_env_config, 'horizon', policy.diffusion_model.horizon)
        self.horizon = horizon
        print(f'[plan_an_env] horizon', horizon)
        print('obs_selected_dim:', self.obs_selected_dim)

        # list of numpy, extract start and target
        obs_start_list, target_list = kuka_obs_target_list(self.problems_dict, self.maze_idx, self.prob_permaze)

        ## -------------- Convert start and goal locations to torch for model input ---------------

        ## arr here is actually tensor
        obs_start_arr, target_arr = self.fit_obs_target_tensor(obs_start_list, target_list, self.samples_perprob)
        prob_start_idx = 0
        # first np problems of self.maze_idx: [n_p, n_w*3]
        wall_locations = self.get_wallLoc_tensor_single(prob_start_idx, self.prob_permaze, self.samples_perprob)
        ## obs_start_arr: eval_n_times, eval_bsize, 4
        # pdb.set_trace()
        assert len(self.obs_selected_dim) == obs_start_arr.shape[-1]
        assert len(self.obs_selected_dim) == target_arr.shape[-1]
        assert obs_start_arr.ndim == 2 # (B, 7)
        assert target_arr.ndim == 2 # (B, 7)
            
        ## ------------------------------------
        ## --------- do the planning ----------
        ## ------------------------------------

        ## set conditioning xy position to be the goal
        ## cond[0]: start observation; cond[127]: [x,y,0,0]
        # print('target_batch', target_batch.shape) # torch.Size([B, 2])
        ## NOTE cond is for the diffusion model
        cond = {
            0: obs_start_arr, # (n_e*n_s, 7)
            horizon - 1: target_arr,
        }

        ## conditional_sample shape: (B, 128, 6) walls_loc torch.Size([B, 6])
        ## i_batch 0: cond torch.Size([40, 4]) wall_locations torch.Size([40, 6])
        # print(f'i_batch {i_batch}: cond', cond[diffusion.horizon - 1].shape, 'wall_locations', wall_locations.shape)

        other_results = {}
        self.other_results = other_results
        other_results['replan_suc_list'] = [] # list of bool
        other_results['no_replan_suc_list'] = [] # list of bool
        # pdb.set_trace()
        # action: first action; samples: trajectory; batch_size should be 1
        start_time = time.time()
        if type(policy) == PolicyCompose:
            raise NotImplementedError

        elif type(policy) == Policy:
            
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

        self.plan_option = 1
        if self.plan_option == 1:
            opt1_dict = self.option_1_pick_1_traj(samples, env, self.prob_permaze)
            # vis_modelout_path_list = opt1_dict['modelout_path_list'] # a list

        if self.do_replan:
            # will inplaced modified the num_colck_list
            self.replan_pred_trajs(opt1_dict, env, wall_locations, other_results)

        vis_modelout_path_list = opt1_dict['modelout_path_list'] # a list
        # prob_time_list: a list time for check collision and replan of the batch
        other_results['batch_runtime'] += np.array( opt1_dict['prob_time_list'] ).sum()
        other_results['num_colck_list'] = opt1_dict['num_colck_list']

        
        assert vis_modelout_path_list[0].shape[-1] == len(self.obs_selected_dim)


        vis_maze_idxs_list = [self.maze_idx,] * len(vis_modelout_path_list)
        ## vis_modelout_path_list[0].shape # (1, 122, 7)

        return vis_modelout_path_list, vis_maze_idxs_list, other_results
    

    def option_1_pick_1_traj(self, samples, env, prob_permaze):
        '''We plan samples_perprob candidates for each problems,
        now pick the successful one, * all in numpy *
        1. only append the success traj
        2. if no success append the last one
        '''
        # 1. ---- reshape ----
        # samples.observations (np): unnorm [B=n_p*n_s, horizon, 7]
        config_dim =  samples.observations.shape[-1]
        new_shape = (prob_permaze, self.samples_perprob, self.horizon, config_dim)
        ## reshape would be risky, manually checked
        pred_trajs = samples.observations.reshape(*new_shape).copy() # need copy?

        # 2. list to hold results

        modelout_path_list = [None] * prob_permaze # list of np
        replan_candidate_list = [None] * prob_permaze # list of (list of np)
        prob_time_list = [None] * prob_permaze
        num_colck_list = [None] * prob_permaze
        suc_list = [None] * prob_permaze

        # 3. --- collision checker ----
        checker = KukaCollisionChecker(normalizer=None) # already unnormed
        self.replanner.set_collision_checker(checker) # shallow copy
        # pred_trajs (20, 4, 48, 7)
        print('pred_trajs', pred_trajs.shape)

        for i_p in range(prob_permaze):
            
            tic = time.time()
            num_collision_check = 0
            for i_aj in range(self.samples_perprob):
                # 1. check if the traj if good, (*two end* and collision)
                traj = pred_trajs[i_p][i_aj]
                assert type(traj) == np.ndarray and traj.ndim == 2

                # [Caution] *two end* are not checked
                is_valid, num_colck = checker.check_single_traj( traj, env )

                num_collision_check += num_colck
                if is_valid:
                    modelout_path_list[i_p] = traj
                    print(f'prob{i_p} traj{i_aj}: success')
                    suc_list[i_p] = True
                    break
            if not suc_list[i_p]:
                replan_candidate_list[i_p] = pred_trajs[i_p]
            
            prob_time = time.time() - tic
            prob_time_list[i_p] = prob_time if self.samples_perprob > 1 else 0
            num_colck_list[i_p] = num_collision_check
            
            ## all traj fails, use the last one
            if modelout_path_list[i_p] is None:
                modelout_path_list[i_p] = traj
                suc_list[i_p] = False

        print('suc_list', suc_list) # list
        print('suc_list mean', np.array(suc_list).mean())
        # pdb.set_trace()
        opt1_dict = dict(
            modelout_path_list=modelout_path_list,
            replan_candidate_list=replan_candidate_list,
            prob_time_list=prob_time_list,
            num_colck_list=num_colck_list,
            suc_list=suc_list
        )
        return opt1_dict


    




    def replan_pred_trajs(self, opt1_dict, env, wall_locations, other_results):
        '''
        NOTE Inplace modification to modelout_path_list, other_results
        NOTE need to update collision_check here
        NOTE use the last path to replan...
        wall_locations: B, n_w*3
        '''
        modelout_path_list = opt1_dict['modelout_path_list'] # list of (h, 7/14)
        replan_candidate_list = opt1_dict['replan_candidate_list']
        suc_list = opt1_dict['suc_list']
        other_results['no_replan_suc_list'] = suc_list
        other_results['replan_suc_list'] =  suc_list[:] # deepcopy

        assert wall_locations.ndim == 2
        # ------- replan `bsize` trajectories one by one, in CPU, cannot parallel --------- 
        ## loop through all the paths
        for i_p in range(len(modelout_path_list)):
            if suc_list[i_p]:
                continue
            ## not success
            # ----------- replan -------------
            if self.do_replan:
                # also good for seq mode, because wall_locations: (10, 96) all same
                i_w = i_p * self.samples_perprob 
                # a list of np2d or np3d, if seq list of size 1
                traj_candidates = replan_candidate_list[i_p]
                cnt_colck = 0
                assert traj_candidates[0].ndim == 2
                n_repl = min(6, len(traj_candidates)) # hyper param
                tic = time.time()
                for i_aj in range(n_repl):
                    traj = traj_candidates[i_aj]
                    
                    new_traj, is_success, num_colck = \
                        self.replanner.replan_collision_section_less_check(env, traj, 
                                                                           wall_locations[i_w:i_w+1]) # (1,60)
                    cnt_colck += num_colck
                    if is_success:
                        break
                repl_time = time.time() - tic
                
                utils.print_color(f'i_p {i_p} Before:{False} After:{is_success}; chk {num_colck}', c='y')
                other_results['replan_suc_list'][i_p] = is_success # success after replan
                
                opt1_dict['num_colck_list'][i_p] += cnt_colck
                opt1_dict['prob_time_list'][i_p] += repl_time
            # --------------------------------

                if is_success:
                    modelout_path_list[i_p] = new_traj
        
        # NOTE
        opt1_dict['modelout_path_list'] = modelout_path_list





    def fit_obs_target_tensor(self, obs_start_list, target_list, samples_perprob):
        '''
        ## -------------- Convert start and goal locations to torch ---------------
        copy the obs/target correspondingly `samples_perprob` times
        obs_start_list / target_list (a list of numpy or np): should be np
        samples_perprob (int):
        '''
        assert type(obs_start_list) == np.ndarray
        ## B, 1, 7 -> B, ns, 7
        obs_start_arr = torch.tensor(obs_start_list).unsqueeze(1).repeat(1, samples_perprob, 1)
        ## B, 1, 7 -> B, ns, 7
        target_arr = torch.tensor(target_list).unsqueeze(1).repeat(1, samples_perprob, 1)
        
        ## torch.Size([total, 7]) torch.Size([total, 7])
        obs_start_arr = obs_start_arr.flatten(0, 1)
        target_arr = target_arr.flatten(0, 1)
 
        return obs_start_arr, target_arr

    def get_wallLoc_tensor_single(self, prob_start_idx, prob_permaze, samples_perprob):
        '''1. load wallLoc and convert tensor,
           2. broadcast wallLoc samples_perprob times'''

        # ## only use the first 3 wall in case of composing walls ??? maze2d
        ## * in unseen maze: (n_env, n_p, n_w*3), has been preprocessed *
        ## (100(maze), 100(problem), 20, 6)
        ## (2, n_p, 20, 6) -> (n_p, 20, 6)
        # pdb.set_trace()
        wall_locations = self.problems_dict['infos/wall_locations'][self.maze_idx, ]
        wall_locations = wall_locations[prob_start_idx : (prob_start_idx+prob_permaze), ...]
        
        ## For train set: (n_p, 20, 6) -> (n_p, 20, 3)
        if wall_locations.ndim == 3:
            ## might be train set?
            wall_locations = wall_locations[:, :, :3] 
            ## (n_p, 20, 3) -> (n_p, 60)
            wall_locations = wall_locations.reshape(wall_locations.shape[0], -1)
            assert False
        else:
            assert wall_locations.shape[-1] >= 6 # sanity check
        print('LUO wall_locations.shape', wall_locations.shape)

        ## to tensor and copy n_s times
        wall_locations = torch.tensor(wall_locations, dtype=torch.float32)
        ## (n_p, 20, 3) -> (n_p, n_s, 60) -> (np * n_s, 60)
        wall_locations = wall_locations.unsqueeze(1).repeat(1, samples_perprob, 1).flatten(0, 1)

        assert wall_locations.ndim == 2 and wall_locations.shape[1] >= 6

        return wall_locations # e.g. [80， 18], [B, n_w*3]
    
        
    def plan_kuka_save_visfig(self, renderer, maze_prefix, vis_modelout_path_list, \
        vis_maze_idxs_list, plan_env_config):

        ## update instance variable with a dict
        self.update_config(plan_env_config)

        assert self.plan_option == 1, 'only support option 1'

        vis_start_idx = plan_env_config.get('vis_start_idx', 2)
        total_samples = self.prob_permaze * self.samples_perprob
        num_vis = self.num_vis_traj # * samples_perprob # 1
        vis_gap = 1


        ## ---- we have finished 'prob_permaze' problems ------
        assert len(vis_maze_idxs_list) == self.prob_permaze == len(vis_modelout_path_list)

        assert len(np.unique(vis_maze_idxs_list)) == 1, 'should be the same'
        assert type(vis_modelout_path_list[0]) == np.ndarray, 'should be a list of np'
        # assert vis_modelout_path_list[0].ndim == 3 and len(vis_modelout_path_list[0]) == 1


        str_epoch = plan_env_config['str_epoch']

        ## save direct prediction
        fullpath = join(self.savepath_dir, f'{maze_prefix}{self.maze_idx}-modelpred-{str_epoch}.png')
        vis_end_idx = vis_start_idx + num_vis * vis_gap


        if vis_modelout_path_list[0].ndim == 2: # list of 2d (horizon, 7)
            vis_modelout_path_list = pad_and_concatenate_2d(vis_modelout_path_list)
        else:
            raise RuntimeError
        
        # now np (n_p, horizon, 7/14)
        vis_modelout_path_list = vis_modelout_path_list[vis_start_idx:vis_end_idx:vis_gap]
        ## also need to trunc
        vis_maze_idxs_list = vis_maze_idxs_list[vis_start_idx:vis_end_idx:vis_gap] # ?a list of str if composed env?



        renderer.composite(fullpath, vis_modelout_path_list, maze_idx=vis_maze_idxs_list)
    



    def plan_an_env_sequential(self, env, policy, use_normed_wallLoc, plan_env_config:utils.Config):
        """
        Plan one problem each time
        Args:
            env: an env instance
            policy: policy to generate motion plans
            plan_env_config: other planning configurations
        """
        self.update_config(plan_env_config)


        self.return_diffusion = getattr(plan_env_config, 'return_diffusion', False)
        horizon = getattr(plan_env_config, 'horizon', policy.diffusion_model.horizon)
        self.horizon = horizon
        
        print(f'[plan_an_env_sequential] horizon', horizon)


        print('obs_selected_dim:', self.obs_selected_dim)
        # [n_probs, 2] extract start and target
        obs_start_list, target_list = kuka_obs_target_list(self.problems_dict, self.maze_idx, self.prob_permaze)

        
        
        prob_start_idx = 0
        # [np*1, n_w*3] prob_permaze, samples_perprob
        wall_locations = self.get_wallLoc_tensor_single(prob_start_idx, self.prob_permaze, 1)

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
            # pdb.set_trace()

            assert len(self.obs_selected_dim) == obs_start_arr.shape[-1]
            assert len(self.obs_selected_dim) == target_arr.shape[-1]
            assert obs_start_arr.ndim == 2 # (B, 2)
            assert target_arr.ndim == 2 # (B, 2)

            ## NOTE cond is for the diffusion model
            cond = {
                0: obs_start_arr, # (n_e*n_s, 7)
                horizon - 1: target_arr,
            }

            start_time = time.time()
            # action: first action; samples: trajectory; batch_size should be 1    
            if type(policy) == Policy:
                
                samples = policy(cond, batch_size=-1, wall_locations=wloc,use_normed_wallLoc=use_normed_wallLoc, return_diffusion=self.return_diffusion)

                
                samples = samples[1] # (action, traj), we ignore the action part

            else:
                raise NotImplementedError()
            
            end_time = time.time()
        
            self.plan_option = 1
            if self.plan_option == 1:
                self.prob_idx = i_p
                # create opt1_dict, cnt collision
                opt1_dict = self.option_1_pick_1_traj(samples, env, prob_permaze=1)
                assert len(opt1_dict['modelout_path_list']) == 1
                single_out = opt1_dict['modelout_path_list'][0] # np 2d?

            else:
                # pdb.set_trace()
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
        
        ## vis_modelout_path_list[0].shape # (1, 122, 7)
        ## 3 items
        return vis_modelout_path_list, vis_maze_idxs_list, other_results