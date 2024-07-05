import time, sys
sys.path.append('.')
import numpy as np
import argparse
import gym, pdb, traceback, os
import multiprocessing as mp
from pb_diff_envs.utils.val_utils_mp import val_save_to_hdf5, is_val_prob_exist
from pb_diff_envs.utils.gen_val_sm2d_utils import gen_val_m2d_data_mp
from pb_diff_envs.scripts.gen_k7d_list_mp import check_param_list
from pb_diff_envs.environment.static.maze2d_rand_wgrp import Maze2DRandRecGroupList
from tqdm import tqdm
from pb_diff_envs.utils.gen_utils import check_data_len, npify


def reset_data_m2d():
    return {'problems': [],
            'maze_idx': [],
            'infos/wall_locations': [],
            'planning_time': [],
            'n_colchk': [],
            }

def append_data_m2d(data, prob, wall_locations, maze_idx, pl_time, n_colchk):
    """

    """
    data['problems'].append(prob.copy()) # (1, n_p, 2, dim)
    data['maze_idx'].append(maze_idx.copy()) # (1, 1) -> final (ng, 1)
    data['infos/wall_locations'].append(wall_locations.copy()) # (1, n_p, n_w, 3)
    data['planning_time'].append(pl_time.copy())
    data['n_colchk'].append(n_colchk.copy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='kuka7d-luotest', help='kuka7d env id')
    # TODO change default here
    parser.add_argument('--num_samples_permaze', type=int, help='Num problems to collect')
    parser.add_argument('--num_mazes', type=int, default=-1, help='Num of mazes (override env if > 0). Used when create evaluation placeholder data.')
    parser.add_argument('--load_mazeEnv', type=str, default="True", help='')
    parser.add_argument('--num_cores', type=int, default=3, help='')
    parser.add_argument('--num_parts', type=int, default=None, help='') # 150?
    args = parser.parse_args()
    assert not is_val_prob_exist(args) or 'testOnly' in args.env_name
    vis_dir = f'./visualization/{args.env_name}_eval/'
    os.makedirs(name=vis_dir, exist_ok=True)
    
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"),\
                            is_eval=True)
    
    env_dvg_list: Maze2DRandRecGroupList
    ## eval groups
    num_groups = args.num_mazes # num_groups if args.num_mazes <= 0 else args.num_mazes
    assert num_groups <= 1000
    assert args.num_mazes == env_dvg_list.eval_num_groups
    assert args.num_samples_permaze == env_dvg_list.eval_num_probs
    args.n_prob_permaze = args.num_samples_permaze
    print(f'num_samples_permaze: {args.num_samples_permaze}')

    num_groups = env_dvg_list.eval_num_groups
    assert args.num_mazes == num_groups
    assert args.num_samples_permaze  == env_dvg_list.eval_num_probs
    # checkparam not sure if the value is good
    interp_density = env_dvg_list.mazelist_config['interp_density']
    args.vis_every = num_groups // 30 if num_groups >= 30 else 1 #  e.g. 200 -> 20: we vis 10 envs

    num_cores = args.num_cores
    num_parts = args.num_parts
    assert num_groups % args.num_parts == 0 # or num_groups < args.num_parts
    print("num_cores: " + str(num_cores))
    print("num_parts: " + str(num_parts))



    ## 2. ------ create list of params ------
    start = 0
    param_list = []
    each_num = num_groups // num_parts # num_cores, number of envs in each proc
    residual_num = num_groups % num_parts # num_cores # e.g. 2
    assert residual_num == 0 # or residual_num == num_groups
    env_dvg_list.model_list = []
    num_mp = num_parts
    all_wall_locations = np.array(env_dvg_list.wallLoc_list, dtype=np.float32)[:,:,:env_dvg_list.world_dim] # n_env, n_wall, 2



    for pid in tqdm(range(num_mp)):
        if pid == num_parts: # num_cores. last iter, might not reach
            end = start + residual_num
            assert False
        else:
            each_num_tmp = each_num if pid >= residual_num else each_num + 1
            end = start + each_num_tmp
        ## pid, args, env_dvg_list, gen_config)
        gen_config = dict(start_idx=start, 
                          end_idx=end,
                          vis_dir=vis_dir, interp_density=interp_density,
                          # (100, 20, 3), just for checking
                          envs_wallLoc=all_wall_locations[start:end].copy(),
                          rng_val=None,
                          reset_data=reset_data_m2d,
                          append_data=append_data_m2d,)

        param = (pid, args, None, gen_config) ## args will be copy automatically
        param_list.append(param)
        start += each_num_tmp

    check_param_list(param_list, num_groups)
    print(param_list[0], param_list[1], param_list[-2], param_list[-1])

    ## ----- Test one Process -----
    # data = gen_val_m2d_data_mp(*param_list[0], lock=None)
    # npify(data)
    # check_data_len(data, len(data['problems']))
    # pdb.set_trace()
    ## ----------------------------



    ## 3. ------ Create Processes ------
    ## Pool
    # Create a manager to provide shared resources
    manager = mp.Manager()
    lock = manager.Lock() # a shared lock, for Pool

    assert mp.cpu_count() >= num_cores
    with mp.Pool(processes=num_cores) as pool:
        results_mp = []
        for i_mp in range(len(param_list)):
            result = pool.apply_async( gen_val_m2d_data_mp, args=param_list[i_mp]+(lock,) )
            results_mp.append(result)
        results_mp = [p.get() for p in results_mp]


    data = val_aggregate_data_dict_m2d(results_mp)

    val_save_to_hdf5(args, data, env_dvg_list, num_groups, vis_dir) # m2d + k7d

    print('gen static maze2d val problems finished.')


def val_aggregate_data_dict_m2d(data_list):
    data_agg = reset_data_m2d()
    for i in range(len(data_list)):
        data_tmp = data_list[i]
        for k,v in data_tmp.items():
            assert type(data_tmp[k]) == list, f'key: {k}'
            data_agg[k] += data_tmp[k]
    return data_agg

if __name__ == '__main__':
    main()


