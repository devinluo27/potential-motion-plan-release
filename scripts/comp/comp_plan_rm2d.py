import numpy as np
import pdb, sys
sys.path.append('.')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import copy

from diffuser.guides.comp_plan_env import ComposedEnvPlanner

from diffuser.guides.comp_plan_helper import ComposedDiffusionPlanner
from diffuser.utils.comp.comp_setup import ComposedParser
from tap import Tap


class Parser(Tap):
    config: str 
    plan_n_maze: int
    cond_w: list = None

def main(args_comp):
    #------------------------------- setup -----------------------------------#
    cp = ComposedParser(args_comp.config, args_comp)
    args_train_list, args_list = cp.setup_args_list(args_comp, )

    dplanner = ComposedDiffusionPlanner(args_train_list, args_list, cp.comp_dataset)
    rm2d_planner = ComposedEnvPlanner()


    #---------------------------- start planning -----------------------------#

    avg_result_dict = dplanner.plan(args_train_list[0], args_list[0], rm2d_planner)
    
    return avg_result_dict


if __name__ == '__main__':

    args_c = Parser().parse_args()

    n_evals = list( range(1) )

    e_list = []
    for i in n_evals:
        tmp = main( copy.deepcopy(args_c), )
        e_list.append(tmp)

    
    # save_depoch_result(e_list, depoch_list, args)