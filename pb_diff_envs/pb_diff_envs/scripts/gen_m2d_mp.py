import numpy as np
import h5py, argparse
from pb_diff_envs.environment.static.rand_maze2d_env import RandMaze2DEnv
from pb_diff_envs.environment.static.comp_rm2d_env import ComposedRM2DEnv
from pb_diff_envs.environment.dynamic.dyn_rm2d_env import DynamicRandMaze2DEnv
from pb_diff_envs.environment.abs_maze2d_env import AbsDynamicMaze2DEnv
from pb_diff_envs.utils.kuka_utils_luo import SolutionInterp
import gym, pdb, traceback, os
from tqdm import tqdm
from pb_diff_envs.utils.utils import save_gif
import multiprocessing as mp
from pb_diff_envs.utils.file_proc_utils import get_rest_idx
from pb_diff_envs.utils.gen_utils import check_data_len, npify
from pb_diff_envs.utils.gen_static_m2d_utils import gen_static_maze2d_data_mp
from pb_diff_envs.utils.dyn.gen_dm2d_utils import gen_dm2d_data_mp
from pb_diff_envs.scripts.gen_k7d_list_mp import check_wallLoc_matched, check_param_list

def reset_data_m2d():
    return {'observations': [],
            'actions': [], # placeholder, all zero, 1D (n,)
            'terminals': [],
            'rewards': [], # placeholder, all zero
            'timeouts': [], # placeholder, all zero
            # idx in self.model_list, use to retrieve the corresponding env
            'maze_idx': [],
            # if this episode terminated by reaching the max len
            'reach_max_episode_steps': [], # placeholder, all zero
            'infos/goal': [],
            'infos/wall_locations': [],
            'is_sol_kp': [], # if the point is direct return from BIT*
            'pos_reset':[],
            }

def append_data_m2d(data, s, a, tgt, done, wall_locations, maze_idx, reach_max_episode_steps, is_sol_kp, pos_reset):
    """
    wall_locations: numpy array (n_walls, 2)
    """
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(np.zeros_like(done, dtype=np.float32)) # 0.0
    data['terminals'].append(done)
    data['timeouts'].append(np.zeros_like(done, dtype=bool)) # False
    data['maze_idx'].append(maze_idx)
    data['reach_max_episode_steps'].append(reach_max_episode_steps)
    data['infos/goal'].append(tgt)
    data['infos/wall_locations'].append(wall_locations.copy())
    data['is_sol_kp'].append(is_sol_kp)
    data['pos_reset'].append(pos_reset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='kuka7d-luotest', help='kuka7d env id')
    # parser.add_argument('--num_episodes_permaze', type=int, default=int(100), help='')
    # TODO change default here
    parser.add_argument('--num_samples_permaze', type=int, help='Num samples to collect')
    parser.add_argument('--num_mazes', type=int, default=-1, help='Num of mazes (override env if > 0). Used when create evaluation placeholder data.')
    parser.add_argument('--load_mazeEnv', type=str, default="True", help='')
    parser.add_argument('--num_cores', type=int, default=3, help='')
    parser.add_argument('--num_parts', type=int, default=None, help='') # 150?
    parser.add_argument('--gen_rest', type=str, default="False", help='', choices=['True', 'False'])
    parser.add_argument('--reversed', type=str, default="False", help='', choices=['True', 'False'])

    args = parser.parse_args()
    vis_dir = f'./visualization/{args.env_name}/'
    os.makedirs(name=vis_dir, exist_ok=True)
    
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"))
    assert env_dvg_list.samples_per_env == args.num_samples_permaze, 'equal to env'
    num_groups = env_dvg_list.num_groups
    num_groups = num_groups if args.num_mazes <= 0 else args.num_mazes

    interp_density = env_dvg_list.mazelist_config['interp_density'] # checkparam
    n_vis = 25
    args.vis_every = num_groups // n_vis if num_groups >= n_vis else 1 #  e.g. 500 -> 25: we vis 20 envs
    assert env_dvg_list.gen_num_parts == args.num_parts
    # pdb.set_trace()

    num_cores = args.num_cores
    num_parts = args.num_parts if args.num_parts is not None else num_cores
    args.num_parts = num_parts # for file proc
    print("num_cores: " + str(num_cores))
    print("num_parts: " + str(num_parts))
    
    start = 0
    param_list = []
    each_num = num_groups // num_parts # num_cores, number of envs in each proc
    residual_num = num_groups % num_parts # num_cores # e.g. 2
    env_dvg_list.model_list = []
    num_mp = num_parts
    all_wall_locations = np.array(env_dvg_list.wallLoc_list, dtype=np.float32)[:,:,:2] # 15000, 20, 3

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
                          envs_wallLoc=all_wall_locations[start:end].copy(), # (100, 20, 3)
                          reset_data=reset_data_m2d,
                          append_data=append_data_m2d,)

        param = (pid, args, None, gen_config)

        param_list.append(param)
        start += each_num_tmp
    
    check_param_list(param_list, num_groups)
    if len(param_list) >= 2:
        print(param_list[0], param_list[1], param_list[-2], param_list[-1])

    if 'DynrandSmaze2d' in args.env_name:
        func_mp = gen_dm2d_data_mp
    else:
        func_mp = gen_static_maze2d_data_mp

    ## -------- Test one Process --------
    # data = func_mp(*param_list[0], lock=None)
    # npify(data)
    # check_data_len(data, len(data['observations']))
    # pdb.set_trace()
    ## ----------------------------------


    if args.gen_rest.strip() == "True":
        rest_dix = get_rest_idx(args, num_groups, './datasets/cache', args.env_name)
        print('rest_dix:', rest_dix)
        param_list = [param_list[i_p] for i_p in rest_dix]

    if args.reversed.strip() == "True":
        param_list = param_list[::-1]
        # param_list = param_list[:250]


    # param_list = param_list[1000:]
    print(f'len param_list: {len(param_list)}')

    
    use_proc_pool = True
    if use_proc_pool:
        lock = None # useless
        # assert mp.cpu_count() >= num_cores
        with mp.Pool(processes=num_cores) as pool:
            results_mp = []
            for i_mp in range(len(param_list)):
                result = pool.apply_async( func_mp, args=param_list[i_mp]+(lock,) )
                results_mp.append(result)

            results_mp = [p.get() for p in results_mp]



    print('multiprocess all finished.')

    data = aggregate_data_dict_m2d(results_mp)

    save_to_hdf5_sm2d(args, data, env_dvg_list, num_groups, vis_dir)

    exit(0)



def aggregate_data_dict_m2d(data_list):
    data_agg = reset_data_m2d()
    for i in range(len(data_list)):
        data_tmp = data_list[i]
        for k,v in data_tmp.items():
            data_agg[k] += data_tmp[k]
    return data_agg

def save_to_hdf5_sm2d(args, data, env_dvg_list, num_groups, vis_dir, vis_idx=None, is_ee3d=False):
    '''save the whole dataset'''
    prefix = './datasets'
    fname = '%s/%s.hdf5' % (prefix, args.env_name) # noisy is added in env id
    if not type(data['observations']) == np.ndarray:
        npify(data)
    check_data_len(data, len(data['observations']))

    vis_idx = [num_groups - 1,] if vis_idx is None else vis_idx # default last maze
    ## vis several mazes
    for i_v in vis_idx:
        env_last = env_dvg_list.create_single_env(i_v)
        v_start = i_v * env_dvg_list.samples_per_env
        v_end = v_start + 250
        tmp = f'{vis_dir}/vis_{i_v}_last.gif'

        if type(env_last) in [RandMaze2DEnv, ComposedRM2DEnv]:
            env_last.render_1_traj(tmp, data['observations'][v_start:v_end])
        # elif type(env_last) == DynamicRandMaze2DEnv:
        elif isinstance(env_last, AbsDynamicMaze2DEnv):
            env_last.render_1_traj(data['observations'][v_start:v_end], 
                               data['infos/wall_locations'][v_start:v_end],
                               savepath=tmp,
                               )
        else:
            raise NotImplementedError()

    
    if args.gen_rest.strip() == "False":
        check_wallLoc_matched(data, env_dvg_list)

        with h5py.File(fname, 'w') as dataset:
            for k in data:
                dataset.create_dataset(k, data=data[k], compression='gzip')

    if 'testOnly' not in fname:
        os.chmod(fname, 0o444)
    print(f'[save hdf5] {fname}')
    
    wallLoc_list = np.array(env_dvg_list.wallLoc_list)

    npyname = '%s/%s.npy' % (prefix, args.env_name)
    np.save(npyname, wallLoc_list)



if __name__ == '__main__':
    main()
