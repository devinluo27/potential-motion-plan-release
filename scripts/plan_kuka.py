import numpy as np
import pdb, sys
sys.path.append('.')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import diffuser.utils as utils
from diffuser.guides.kuka_plan_utils import save_depoch_result
from datetime import datetime
import os.path as osp
import copy
from diffuser.guides.plan_pb_diff_helper import DiffusionPlanner
from diffuser.guides.kuka_plan_env import KukaEnvPlanner

class Parser(utils.Parser):
    dataset: str = None
    config: str = None

def main(args_train, args):
    #---------------------------------- setup ----------------------------------#

    dplanner = DiffusionPlanner(args_train, args)
    rm2d_planner = KukaEnvPlanner()

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
    start_e = 2e5
    depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()


    args.repl_dn_steps = 3
    args.n_replan_trial = 3

    ## set hyperparam
    if 'dualkuka' not in args.dataset.lower():
        ## Kuka 7D
        args.horizon = 52
        args.ddim_steps = 8
        depoch_list = [int(latest_e),] # manully set epoch here, e.g. int(19e5)
        args.seq_eval = False # True: evaluate motion planning problem one by one
        args.use_ddim = True 
        args.load_unseen_maze = True
        args.cond_w = 2.0 # 2.0
        args.do_replan = False
        args.n_prob_env = 20
        args.n_vis = 0 # how many git visualizations for each env
        args.vis_start_idx = 0
        args.samples_perprob = 20
    else:
        ## dualkuka 14D
        args.horizon = 56
        args.ddim_steps = 10

        depoch_list = [int(latest_e),]
        args.seq_eval = False
        args.use_ddim = True
        args.load_unseen_maze = True
        args.cond_w = 2.0
        args.do_replan = False # False # True
        args.n_prob_env = 20
        args.n_vis = 0 # 
        args.vis_start_idx = 3
        args.samples_perprob = 20
        


    sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S")}-nm{int(args.plan_n_maze)}'
    args.savepath = osp.join(args.savepath, sub_dir)

    e_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        e_list.append(tmp)
    

    save_depoch_result(e_list, depoch_list, args)