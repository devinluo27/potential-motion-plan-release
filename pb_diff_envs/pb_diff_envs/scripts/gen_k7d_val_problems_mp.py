from pb_diff_envs.environment.static.rand_kuka_env import RandKukaEnv
import numpy as np
from importlib import reload
import argparse
# from pb_diff_envs.objects.static.voxel_group import DynVoxelGridGroupList
import gym, pdb, traceback, os
from pb_diff_envs.utils.utils import save_gif
import multiprocessing as mp
from pb_diff_envs.utils.val_utils_mp import gen_val_kuka_data_mp, val_save_to_hdf5, is_val_prob_exist
from pb_diff_envs.scripts.gen_k7d_list_mp import check_param_list
from tqdm import tqdm



def reset_data():
    return {'problems': [],
            'maze_idx': [],
            'infos/wall_locations': [],
            'planning_time': [],
            'n_colchk': [],
            }

def append_data(data, prob, wall_locations, maze_idx, pl_time, n_colchk):
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
    # parser.add_argument('--num_episodes_permaze', type=int, default=int(100), help='')
    # TODO change default here
    parser.add_argument('--num_samples_permaze', type=int, help='Num problems to collect')
    parser.add_argument('--num_mazes', type=int, default=-1, help='Num of mazes (override env if > 0). Used when create evaluation placeholder data.')
    parser.add_argument('--load_mazeEnv', type=str, default="True", help='')
    parser.add_argument('--num_cores', type=int, default=3, help='')
    parser.add_argument('--num_parts', type=int, default=None, help='') # 150?

    ## no_check_bit
    ## True: do not check if a solution exists for the problem
    ## False: use BIT* to check if a solution exists for the problem
    parser.add_argument('--no_check_bit', type=str, default="False", choices=['True', 'False'], help='')
    parser.add_argument('--npi', type=int, default=0, help='') # 150?

    args = parser.parse_args()
    args.no_check_bit = args.no_check_bit == 'True'
    
    # assert not is_val_prob_exist(args), 'validation problem set already exists'
    vis_dir = f'./visualization/{args.env_name}_eval/'
    os.makedirs(name=vis_dir, exist_ok=True)
    
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"),\
                            is_eval=True)
    
    # e.g., env_dvg_list: DualKuka_VoxelRandGroupList
    ## eval groups
    num_groups = args.num_mazes # num_groups if args.num_mazes <= 0 else args.num_mazes
    if not args.no_check_bit:
        assert num_groups <= 1000
        assert args.num_mazes == env_dvg_list.eval_num_groups
        assert args.num_samples_permaze == env_dvg_list.eval_num_probs
    else:
        assert args.npi != 0

    args.n_prob_permaze = args.num_samples_permaze
    print(f'num_samples_permaze: {args.num_samples_permaze}')


    num_groups = env_dvg_list.eval_num_groups

    # checkparam not sure if the value is good
    interp_density = env_dvg_list.mazelist_config.get('interp_density', 3.0)
    args.vis_every = num_groups // 30 if num_groups >= 30 else 1 #  e.g. 200 -> 20: we vis 10 envs

    num_cores = args.num_cores
    num_parts = args.num_parts

    
    assert num_groups % args.num_parts == 0 ## or args.num_mazes > 0
    assert num_groups == args.num_mazes
    print("num_cores: " + str(num_cores))
    print("num_parts: " + str(num_parts))

    # rng_val = np.random.default_rng(env_dvg_list.rng_seed) # ?

    ## 2. ------ create list of params ------
    start = 0
    param_list = []
    # assert num_groups % num_cores == 0, f'num_groups:{num_groups}' # outdated delete
    each_num = num_groups // num_parts # num_cores, number of envs in each proc
    residual_num = num_groups % num_parts # num_cores # e.g. 2
    env_dvg_list.model_list = []
    # num_mp = num_parts + 1 if residual_num else num_parts #  num_cores + 1, num_cores
    num_mp = num_parts
    all_wall_locations = np.array(env_dvg_list.wallLoc_list, dtype=np.float32)[:,:,:3] # n_env, n_wall, 3


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
                          no_check_bit=args.no_check_bit,
                          reset_data=reset_data,
                          append_data=append_data,)

        param = (pid, args, None, gen_config) ## args will be copy automatically

        param_list.append(param)
        start += each_num_tmp

    check_param_list(param_list, num_groups)
    print(param_list[0], param_list[1], param_list[-2], param_list[-1])





    ## 3. ------ Create Process ------
    # Create a manager to provide shared resources
    manager = mp.Manager()
    lock = manager.Lock() # a shared lock, for Pool

    assert mp.cpu_count() >= num_cores, 'sanity check if resources are enough'
    with mp.Pool(processes=num_cores) as pool:
        results_mp = []
        for i_mp in range(len(param_list)):
            result = pool.apply_async( gen_val_kuka_data_mp, args=param_list[i_mp]+(lock,) )
            results_mp.append(result)
        results_mp = [p.get() for p in results_mp]



    data = val_aggregate_data_dict(results_mp)

    val_save_to_hdf5(args, data, env_dvg_list, num_groups, vis_dir)


def val_aggregate_data_dict(data_list):
    data_agg = reset_data()
    for i in range(len(data_list)):
        data_tmp = data_list[i]
        for k,v in data_tmp.items():
            assert type(data_tmp[k]) == list, f'key: {k}'
            data_agg[k] += data_tmp[k]
    return data_agg

if __name__ == '__main__':
    main()


