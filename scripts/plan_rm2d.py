import numpy as np
import pdb, sys
sys.path.append('.')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import diffuser.utils as utils

from diffuser.guides.kuka_plan_utils import save_depoch_result
from datetime import datetime
import os.path as osp
import copy

from diffuser.guides.plan_pb_diff_helper import DiffusionPlanner
from diffuser.guides.rm2d_plan_env import RandMaze2DEnvPlanner



class Parser(utils.Parser):
    dataset: str = None
    config: str


def main(args_train, args):
    
    #---------------------------------- setup ----------------------------------#

    dplanner = DiffusionPlanner(args_train, args)
    rm2d_planner = RandMaze2DEnvPlanner()

    #---------------------------- start planning -----------------------------#

    avg_result_dict = dplanner.plan(args_train, args, rm2d_planner)
    
    return avg_result_dict


if __name__ == '__main__':
    ## training args
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')

    ## 1. get epoch to eval on, by default all
    loadpath = args.logbase, args.dataset, args_train.exp_name,
    latest_e = utils.get_latest_epoch(loadpath)
    n_e = round(latest_e // 1e5) + 1 # all
    start_e = 5e5; # 2e5 end_e = 
    depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()


    # args.horizon = # default 48
    args.repl_dn_steps = 3
    args.n_replan_trial = 3

    if 'Dyn' not in args.dataset:
        ## Static Maze2D
        args.seq_eval = False
        depoch_list = [int(19e5),] # int(5e5),]
        args.ddim_steps = 8

        args.n_prob_env = 20
        args.use_ddim = True # False # True
        args.load_unseen_maze = True
        args.cond_w = 2.0 # 2.0
        args.do_replan = False # False # True
        args.n_vis = 20 if args.do_replan else 20
        args.vis_start_idx = 0
        args.samples_perprob = 10
    else:
        ## Dynamic rm2d
        args.seq_eval = False
        depoch_list = [int(19e5),]
        args.horizon = 48
        args.ddim_steps = 8
        
        args.n_prob_env = 20
        args.use_ddim = True
        args.load_unseen_maze = True
        args.cond_w = 2.0
        args.do_replan = False ## replan not supported yet
        args.n_vis = 0 # 2
        args.vis_start_idx = 0
        args.samples_perprob = 20


    sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S")}-nm{int(args.plan_n_maze)}'
    args.savepath = osp.join(args.savepath, sub_dir)

    e_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        e_list.append(tmp)
    
    # if not hasattr(args, 'comp_dataset'):
        # save_depoch_result(e_list, depoch_list, args)