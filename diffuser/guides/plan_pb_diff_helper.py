import numpy as np
import pdb
import torch
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.guides.kuka_plan_utils import plan_kuka_permaze_stat, load_eval_problems_pb, plan_kuka_save_per_env_stat, compute_kuka_avg_result, plan_pbdiff_save_all_stat
from diffuser.datasets.preprocessing import rm2d_kuka_preproc, dyn_rm2d_val_probs_preproc, preproc_val_srm2dwloc_43
from diffuser.guides.rm2d_plan_env import RandMaze2DEnvPlanner, RM2DReplanner
from diffuser.guides.kuka_replan import KukaDiffusionReplanner
from datetime import datetime
import os.path as osp
from diffuser.utils import KukaRenderer
from diffuser.utils.rm2d_render import RandStaticMazeRenderer


class DiffusionPlanner:
    '''can be used for rm2d and kuka, not composing'''
    def __init__(self, args_train, args) -> None:
        self.args_train = args_train
        self.args = args
        
        self.setup_basic(args_train, args,)
        self.setup_env_list(args_train, args,)
        self.setup_model(args_train, args,)
        self.setup_others(args_train, args,)


    def setup_basic(self, args_train, args,):
        sub_dir = f'e{args.diffusion_epoch // 10000}-{datetime.now().strftime("%y%m%d-%H%M%S")}-nm{int(args.plan_n_maze)}'

        if hasattr(args, 'npi'):
            sub_dir += f'-npi{args.npi}'
        if hasattr(args, 'ddim_eta'):
            sub_dir += f'-eta{args.ddim_eta}'
        
        seq = '-seq' if args.seq_eval else '-batch'
        args.savepath += seq

        args.savepath = osp.join(args.savepath, sub_dir)

        ## load dataset here, dataset is a string: name of the env
        print('args.dataset', type(args.dataset), args.dataset)
        self.use_normed_wallLoc = args_train.dataset_config.get('use_normed_wallLoc', False)
        self.load_unseen_maze = args.load_unseen_maze
        
        ## use the trained env or eval env
        if args.load_unseen_maze:
            self.maze_prefix='us'
        else:
            self.maze_prefix='s'
    

    def setup_env_list(self, args_train, args,):
        self.train_env_list = datasets.load_environment(args.dataset, is_eval=self.load_unseen_maze)
        self.train_normalizer = utils.load_datasetNormalizer(self.train_env_list.dataset_url, 
                                                             args_train, self.train_env_list)
        
        ## use interchangeably here, but basically use different maze seed with the training datasets.
        self.plan_env_list = self.train_env_list
        
        if hasattr(args, 'comp_dataset'):
            tmp_split = args.savepath.split('/')
            tmp_split[1] = osp.join('Compose', args.comp_dataset)
            args.savepath = '/'.join(tmp_split)
            # pdb.set_trace()
            # checkparam luotest
            self.plan_env_list = datasets.load_environment(args.comp_dataset, is_eval=self.load_unseen_maze)

    def setup_model(self, args_train, args,):
        # 1. load
        ld_config = dict(env_instance=self.train_env_list) 
        diffusion_experiment = utils.load_potential_diffusion_model(args.logbase, args.dataset, \
                args_train.exp_name, epoch=args.diffusion_epoch, ld_config=ld_config)
        self.diffusion_experiment = diffusion_experiment
        
        # 2. extract
        diffusion = diffusion_experiment.ema # horizon is the training one
        ## diffusion.horizon = 32 # set another horizon
        self.diffusion = diffusion
        self.epoch = diffusion_experiment.epoch
        # diffusion.debug_mode = True ## Check debug mode
        ## by default they are the same, the is_eval is set to True
        if self.train_env_list == self.plan_env_list:
            self.renderer = diffusion_experiment.renderer
        else:
            raise NotImplementedError()
            # to run ViT baseline
            if 'kuka' in self.plan_env_list.env_type.lower():
                self.renderer = KukaRenderer(self.plan_env_list, is_eval=True)
            else:
                self.renderer = RandStaticMazeRenderer(self.plan_env_list)
                

        diffusion.condition_guidance_w = args.cond_w ## 1.8, 4.0 0.0 checkparam careful
        diffusion.ddim_num_inference_steps = args.ddim_steps
        diffusion.horizon = args.horizon


        self.maze_prefix += f'-cgw{diffusion.condition_guidance_w}-h{diffusion.horizon}-ds{args.ddim_steps}-'

        self.policy = Policy(diffusion, self.train_normalizer, use_ddim=args.use_ddim)

        ## Pick a replanner
        repl_config = dict(use_ddim=args.use_ddim, dn_steps=args.repl_dn_steps, n_replan_trial=args.n_replan_trial)
        if self.train_env_list.world_dim == 3:
            self.replanner = KukaDiffusionReplanner(diffusion, self.train_normalizer, 
                                           repl_config, device=args.device)
        else:
            self.replanner = RM2DReplanner(diffusion, self.train_normalizer, repl_config, args.device)




    def setup_others(self, args_train, args,):
        
        ## Sanity env type check
        assert 'maze2d' in self.train_env_list.env_type.lower() or \
                    'kuka' in self.train_env_list.env_type.lower()

        pd_config = {'load_unseen_maze': args.load_unseen_maze, 
                     'horizon': args_train.horizon}

        self.problems_dict = load_eval_problems_pb(self.plan_env_list, pd_config) # to check

        tmp = tuple(range(self.train_env_list.world_dim)) # (0,1,2)
        
        # if 'C43' in self.train_env_list.env_name:
        if 'concave' in self.train_env_list.env_type.lower():
            
            n_probs = self.problems_dict['infos/wall_locations'].shape[1]
            new_wloc = preproc_val_srm2dwloc_43( self.train_env_list, n_probs)
            self.problems_dict['infos/wall_locations'] = new_wloc
            tmp = (0, 1, 2)

        # illustration, e.g. for kuka:
        # (n_envs, n_probs, 2, 7)
        # probs = problems_dict['problems'] # 4d array of planning problems
        # (n_envs, n_probs, n_walls, loc+size=6) -> [n_envs, n_probs,n_walls*size] in preproc
        # wall_loc_list = problems_dict['infos/wall_locations'] 

        if 'dyn' in self.train_env_list.env_type.lower():
            ## dyn_maze2d
            dyn_rm2d_val_probs_preproc(self.problems_dict, wloc_select=tmp, env=self.train_env_list)
        else:
            rm2d_kuka_preproc(self.problems_dict, wloc_select=tmp, env=self.train_env_list) # trimed/flatten wall_loc

        


    def plan(self, args_train, args, rm2d_planner):
        result_dict_list = []

        problems_dict = self.problems_dict # should be already preprocessed
        # (n_envs, n_probs, 2, 7)
        # probs = problems_dict['problems'] # 4d array of planning problems
        n_envs, n_prob_env  = problems_dict['problems'].shape[:2]
        replanner = self.replanner # RM2DReplanner, 


        ## ------------- plan/test hyper-parameters -------------------
        do_replan = args.do_replan # True # False
        test_n_envs = int(args.plan_n_maze)
        # n_prob_env # 80! 10 unique traj permaze
        prob_permaze = args.n_prob_env # if not do_replan else 20 # 10, simply takes the first X problems
        assert test_n_envs <= n_envs and prob_permaze <= n_prob_env
        samples_perprob = args.samples_perprob

        n_vis = args.n_vis # 10 # checkparam 2
        vis_start_idx = args.vis_start_idx # 0 # checkparam 2
        str_epoch = f'e{int(self.epoch)//10000}w-s{samples_perprob}-replan{do_replan}'

        for i_e in range(test_n_envs):
            
            env = self.plan_env_list.create_single_env(i_e)
            env.load(GUI=False)
            env.seed(100)
            if i_e > 0: # 
                self.plan_env_list.model_list[i_e-1].unload_env()



            plan_env_config = utils.Config(
                None,
                prob_permaze=prob_permaze,
                samples_perprob=samples_perprob,
                env_mazelist=self.plan_env_list,
                vis_freq=args.vis_freq, # int

                wall_locations_list=None, # *** will use the wallloc from env if not comp

                maze_idx=i_e,  # ***
                obs_selected_dim=args_train.dataset_config['obs_selected_dim'],
                use_waypoint_controller=False,


                ## can add customized horizon.
                ## for save_fig
                str_epoch=str_epoch,
                epoch=self.epoch,
                num_vis_traj=n_vis,
                savepath_dir=args.savepath,
                n_timesteps=None, # seems not used self.diffusion.n_timesteps,
                seed_maze=99,
                n_maze=int(args.plan_n_maze),
                ## vis
                return_diffusion=False,
                num_gif_traj=2,

                problems_dict=problems_dict,
                vis_start_idx=vis_start_idx,

                do_replan=do_replan,
                replanner=replanner,

                use_ddim=args.use_ddim,

                horizon_wtrajs=args_train.horizon, # input w

                js_horizon_pl_list=getattr(args, 'horizon_pl_list', None),
                js_repl_dn_steps=getattr(args, 'repl_dn_steps', None),
                js_ddim_compute_prev_ts=getattr(args, 'ddim_compute_prev_ts', None),

            )

            self.custom_plan_postproc(args, plan_env_config)

            env.load(GUI=False) # replan need loaded env


            if args.seq_eval:
                vis_modelout_path_list, vis_maze_idxs_list, other_results = \
                    rm2d_planner.plan_an_env_sequential(env, self.policy, self.use_normed_wallLoc, plan_env_config)
            else:
                vis_modelout_path_list, vis_maze_idxs_list, other_results = \
                    rm2d_planner.plan_an_env(env, self.policy, self.use_normed_wallLoc, plan_env_config)



            utils.mkdir(args.savepath)

            ## ---------- save visualization ----------
            if n_vis > 0:
                    rm2d_planner.plan_kuka_save_visfig(self.renderer, self.maze_prefix, vis_modelout_path_list, \
                            vis_maze_idxs_list, plan_env_config)
            else:
                print('no visualization')

            env.load(GUI=False) # NOTE: load again
            # --- pass wtrajs for collision check, in dynamic env ----
            # problems_dict: dict_keys(['infos/wall_locations', 'problems', ...])
            # save the problems to env instance
            env.problems_dict = extract_prob_dict(problems_dict, i_e, prob_permaze)


            result_dict = plan_kuka_permaze_stat(vis_modelout_path_list, other_results, env)


            result_dict_list.append(result_dict)
            env.unload_env()


        ## ------- save result as a json file ----------
        ## ---------------------------------------------
        avg_result_dict = compute_kuka_avg_result(result_dict_list)
        plan_kuka_save_per_env_stat(result_dict_list, plan_env_config)
        plan_pbdiff_save_all_stat(avg_result_dict, self.maze_prefix, plan_env_config)

        ## clean up connection
        env.unload_env()

        return avg_result_dict
    
    def custom_plan_postproc(self, args, plan_env_config):
        pass


def extract_prob_dict(inp_dict, i_e, prob_permaze):
    ret = {} # problems: (100, 40, 2, 2); w: (100, 40, 96)
    for k in inp_dict:
        ret[k] = inp_dict[k][i_e, :prob_permaze] # 20, h*nw*d
    return ret
