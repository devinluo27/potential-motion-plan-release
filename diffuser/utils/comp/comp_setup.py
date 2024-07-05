import importlib, os, sys
import copy
import diffuser.utils as utils
from datetime import datetime
import os.path as osp
import numpy as np
import copy, pdb

class Parser(utils.Parser):
    config: str

class ComposedParser:
    '''
    setup hyperparameters for compositional evaluation
    '''
    def __init__(self, config: str, args_cmd=None):
        config = config[:-3]
        self.comp_config = config
        sys.path.append(os.path.dirname(config))
        self.module = importlib.import_module(os.path.basename(config))
        self.mconfig_list = self.module.base['config_list']
        self.comp_dataset: str = self.module.base['dataset']
        self.comp_type: str = self.module.base['comp_type']
        assert self.comp_type in ['maze2d:static+static', 'kuka:static+static', \
                                  'dualkuka:static+static', 'maze2d:dyn+static', ]
        self.n_models = len(self.mconfig_list)
        self.args_cmd = args_cmd
        
    
    def setup_args_list(self, args_sh): # 
        args_train_list = []
        args_list = []
        for i in range(self.n_models):

            args_tmp = copy.deepcopy(args_sh)
            args_tmp.config =  self.mconfig_list[i]
            args_train = Parser().parse_args('diffusion', not_parse=True, input_args=args_tmp)

            args_tmp = copy.deepcopy(args_sh)
            args_tmp.config =  self.mconfig_list[i]
            args = Parser().parse_args('plan', not_parse=True, input_args=args_tmp)

            args.n_prob_env = 20 # 200
            args.ddim_steps = self.module.base['ddim_steps']
            args.repl_dn_steps = 5

            ## TODO: Set Your HyperParameters Here
            ## static rm2D
            if self.comp_type == 'maze2d:static+static':
                ## for static rm2d
                args.ddim_steps = 24 # 10
                args.ddim_compute_prev_ts = args.ddim_steps
                args.ddim_eta = 1.0
                args.repl_dn_steps = 3 # 5
                args.do_replan = False # False or True

                args.uncond_base_idx = 0
                args.seq_eval = False # False # True
                # args.horizon_pl_list = [40, 48, 56, 64] # [48,] # ori design
                args.horizon_pl_list = [36, 40, 44, 48, 52, 56, 60, 64] # [48,]
                # depoch_list = [int(18e5), int(18e5)] # int(9e5)
                # args.horizon_pl_list = [36, 40, 44, 48, 52, 56, 60, 64] # Jan 25
                depoch_list = [int(19e5),] # int(19e5)

                args.use_ddim = True # True # False
                args.load_unseen_maze = True

                if getattr(self.args_cmd, 'cond_w', None) is not None:
                    args.cond_w = [float(ii) for ii in self.args_cmd.cond_w]
                else:
                    args.cond_w = [2.0, 2.0]
                    
                args.n_vis = 20 ## 0
                args.vis_start_idx = 0
                args.samples_perprob = 20
            

            # ------------ Kuka7D -----------
            # elif 'kuka7d' in self.comp_dataset:
            elif self.comp_type == 'kuka:static+static':
                args.repl_dn_steps = 3
                args.ddim_steps = 10
                args.ddim_compute_prev_ts = args.ddim_steps

                # args.no_check_bit = True
                # args.npi = 5
                args.ddim_eta = 1.0

                args.uncond_base_idx = 0
                args.seq_eval = False # False # True
                # static
                # args.horizon_pl_list = [32, 40, 48, 56, 64] # [32, 40, 48, 56] # [48, 64] #
                # args.horizon_pl_list =  [32, 36, 40, 44, 48, 52, 56, 64] # [32, 40, 48, 56] # [48, 64] # Sep 2023
                args.horizon_pl_list =  [36, 40, 44, 48, 52, 56, 60, 64] # Jan 21, 2024
                # args.horizon_pl_list =  [48,]
                depoch_list = [int(18e5), int(18e5)] # int(9e5)
                # depoch_list = [int(19e5), int(19e5)] # int(9e5)
                # depoch_list = [int(latest_e), ] # int(9e5)
                ## no replan
                args.use_ddim = True # True # False # True
                args.load_unseen_maze = True
                # args.cond_w = [2.0, 0.0001] # [3.5, 3.5] # 2.0 # [1.0, 1.0] not good,
                # args.cond_w = [2.0, 0.0] # [3.5, 3.5] # 2.0 # [1.0, 1.0] not good,
                # args.cond_w = [1.5, 1.5] #
                args.cond_w = [2.0, 2.0] #

                # args.cond_w = [2.0, -1.0] #
                # args.cond_w = [1.8, 1.8] #
                # args.cond_w = [1.9, 1.9] # [3.5, 3.5] # 2.0 # [1.0, 1.0] not good,
                args.do_replan = True # False # True # looks like replan does not help when very difficult
                args.n_vis = 1 # 20 # 2
                args.vis_start_idx = 0
                args.samples_perprob = 20 # 20 10 5
            


            # ------------ DualKuka14D -----------
            # elif 'kuka14d' in self.comp_dataset:
            elif self.comp_type == 'dualkuka:static+static':
                args.uncond_base_idx = 0
                args.seq_eval = False # True # False
                # [32, 36, 40, 44, 48, 52, 56, 60, 64]
                # args.horizon_pl_list =  [32, 40, 48, 56, 64] # [32, 40, 48, 56] # [48, 64] #
                args.ddim_eta = 1.0
                # args.horizon_pl_list =  [32, 36, 40, 44, 48, 52, 56, 64] # [32, 40, 48, 56] # [48, 64] #
                args.horizon_pl_list =  [36, 40, 44, 48, 52, 56, 60, 64] # Jan 21
                # args.horizon_pl_list = [48,]#

                args.repl_dn_steps = 3
                args.ddim_steps = 20 # 10 is better than 24
                args.ddim_compute_prev_ts = args.ddim_steps
                # args.no_check_bit = True
                # args.npi = 11

                # depoch_list = [int(19e5), int(19e5)] # int(9e5)
                depoch_list = [int(19e5),] # int(9e5)
                ## no replan
                args.use_ddim = True # True # False # True
                args.load_unseen_maze = True
                args.cond_w = [2.0, 2.0] # 
                # args.cond_w = [2.0, 0.0] #
                # args.cond_w = [2.0, 0.0001] #
                args.cond_w = [float(ii) for ii in self.args_cmd.cond_w]
                args.do_replan = False # True # False # True
                args.n_vis = 0 #20 # 20 # 2
                args.vis_start_idx = 0
                args.samples_perprob = 20 # 5



            # ------------ Dynamic 2d -----------
            elif self.comp_type == 'maze2d:dyn+static':
                args.uncond_base_idx = 1 # static model as uncond
                args.seq_eval = True # False

                args.ddim_eta = 1.0 # 0.0
                args.ddim_steps = 20 # 14 # 14
                args.ddim_compute_prev_ts = args.ddim_steps

                ## Dynamic
                args.horizon_pl_list = [48, 52, 56, 60, 64]# 60:91.048,]
                args.horizon_pl_list = [48, 52, 56, 60, 64, 68, 72]# 60:91.048,] # start 1800
                args.horizon_pl_list = [40, 48, 52, 56, 60, 64, 68]# 60:91.048,] # start 1800
                args.horizon_pl_list = [32, 40, 48, 52, 56, 60, 64, 68]# Jan 31
                args.horizon_pl_list = [32, 40, 48, 52, 56, 60, 64]# Jan 31
                # args.horizon_pl_list = [48,] # 52, Jan23 2024
                depoch_list = [int(19e5),]
                args.use_ddim = True
                args.load_unseen_maze = True
                args.cond_w = [2.0, 2.0] # 2.0
                args.do_replan = False # False # True
                args.n_vis = 0 # 5 # 2 # if args.do_replan else 0
                args.vis_start_idx = 10 # 10
                args.samples_perprob = 20 # 10:91,5:86 4

            else:
                raise NotImplementedError()


            if True: # len(depoch_list) >= 2:
                if not hasattr(args, 'npi'):
                    args.npi = 0
                # sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S")}-nm{int(args.plan_n_maze)}'
                # args.savepath = osp.join(args.savepath, sub_dir)
                
                # custom composed save dir
                tmp_cfg = os.path.basename( self.comp_config )
                args.savepath = os.path.join(args.logbase, 'Compose', self.comp_dataset, tmp_cfg)
                os.makedirs(args.savepath, exist_ok=True)
                # pdb.set_trace()

            # ----------  LuoTest ends ----------

            args_train.diffusion_epoch = depoch_list[0]
            args.diffusion_epoch = depoch_list[0]





            args_train_list.append(args_train)
            args_list.append(args)
        
        return args_train_list, args_list
