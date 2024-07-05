import numpy as np
import torch, pdb
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.guides.kuka_plan_utils import load_eval_problems_pb
from diffuser.datasets.preprocessing import rm2d_kuka_preproc, dyn_rm2d_val_probs_preproc
from diffuser.guides.policies import Policy
from diffuser.guides.policies_compose import PolicyCompose

from diffuser.guides.comp_plan_env import ComposedRM2DReplanner
import os.path as osp
from .plan_pb_diff_helper import DiffusionPlanner
from diffuser.utils.rm2d_render import RandStaticMazeRenderer, RandDynamicMazeRenderer


class ComposedDiffusionPlanner(DiffusionPlanner):
    def __init__(self, args_train_list, args_list, comp_dataset):
        self.n_models = len(args_list)
        self.args_train_list = args_train_list
        self.args_list = args_list
        self.comp_dataset = comp_dataset

        self.setup_basic(args_train_list[0], args_list[0],)
        self.setup_env_list(args_train_list, args_list,)
        self.setup_model(args_train_list, args_list,)
        self.setup_others(args_train_list[0], args_list[0],)


    def setup_env_list(self, args_train_list, args_list,):
        self.train_env_list_list = []
        self.train_normalizer_list = []
        for i_m in range(self.n_models):
            args_train = args_train_list[i_m]
            args = args_list[i_m]
            train_env_list = datasets.load_environment(args.dataset, is_eval=self.load_unseen_maze)
            train_normalizer = utils.load_datasetNormalizer(train_env_list.dataset_url, 
                                                             args_train, train_env_list)
            self.train_env_list_list.append( train_env_list )
            self.train_normalizer_list.append( train_normalizer )

        utils.print_color(self.comp_dataset, c='r')
        self.comp_env_list = datasets.load_environment(self.comp_dataset, is_eval=self.load_unseen_maze)
        self.plan_env_list = self.comp_env_list
        

    def setup_model(self, args_train_list, args_list,):
        '''
        load two models
        '''
        self.diffusion_experiment_list = []
        self.diffusion_list = []
        for i_m in range(self.n_models):
            args_train = args_train_list[i_m]
            args = args_list[i_m]
            # 1. load
            ld_config = dict(env_instance=self.train_env_list_list[i_m]) 
            diffusion_experiment = utils.load_potential_diffusion_model(args.logbase, args.dataset, \
                    args_train.exp_name, epoch=args.diffusion_epoch, ld_config=ld_config)
            
            
            self.diffusion_experiment_list.append( diffusion_experiment )


            # 2. extract
            diffusion = diffusion_experiment.ema # horizon is the training one
            diffusion.condition_guidance_w = args.cond_w[i_m] ## 1.8, 4.0 0.0 checkparam careful

            diffusion.ddim_num_inference_steps = args.ddim_compute_prev_ts
            # diffusion.horizon = 60
            self.diffusion_list.append ( diffusion ) # the model




        
        # ------------ General Setup --------------

        self.epoch = diffusion_experiment.epoch

        # diffusion.debug_mode = True ## 

        # self.renderer = diffusion_experiment.renderer # ????
        if 'Dyn' in self.comp_env_list.env_name:
            self.renderer = RandDynamicMazeRenderer(self.comp_env_list)
        else:
            if 'kuka' in self.comp_env_list.env_name:
                self.renderer = utils.KukaRenderer(self.plan_env_list, is_eval=True)
            else:
                self.renderer = RandStaticMazeRenderer(self.comp_env_list, )

        # args here actually is the args of the last model
        self.maze_prefix += f'-cgw{args.cond_w[0]}x{args.cond_w[1]}-h{diffusion.horizon}-ds{args.ddim_steps}-'
        

        self.policy = PolicyCompose(self.diffusion_list, 
                                    self.train_normalizer_list, 
                                    self.use_normed_wallLoc,
                                    args.use_ddim,
                                    po_config=dict(
                                        num_walls_c=self.comp_env_list.num_walls_c,
                                        wall_dim=self.comp_env_list.world_dim,
                                        comp_type=self.comp_env_list.comp_type,
                                        ddim_steps=args.ddim_steps,
                                        wall_is_dyn=getattr(self.comp_env_list, 'wall_is_dyn', None),
                                        uncond_base_idx=args.uncond_base_idx,
                                        ddim_eta=getattr(args, 'ddim_eta', 0.0),
                                        ),
                                    )
        
        repl_config = dict(use_ddim=args.use_ddim, dn_steps=args.repl_dn_steps)

        self.replanner = ComposedRM2DReplanner(self.policy, self.train_normalizer_list[0], repl_config, args.device)
        



    def setup_others(self, args_train, args,):
        
        assert 'Comp' in str(self.comp_env_list), 'set up the compositional setting'

        pd_config = {'load_unseen_maze': args.load_unseen_maze, 
                     'horizon': args_train.horizon}

        self.problems_dict = load_eval_problems_pb(self.comp_env_list, pd_config)

        tmp = tuple(range(self.comp_env_list.world_dim)) # (0,1,2)
        if 'Dyn' in self.comp_env_list.env_name:
            dyn_rm2d_val_probs_preproc(self.problems_dict, wloc_select=tmp, env=self.comp_env_list)
        else:
            rm2d_kuka_preproc(self.problems_dict, wloc_select=tmp, env=self.comp_env_list) # trimed/flatten wall_loc

        # wall (300, 200, 18)

    def custom_plan_postproc(self, args, plan_env_config):
        plan_env_config._dict['horizon_pl_list'] = args.horizon_pl_list