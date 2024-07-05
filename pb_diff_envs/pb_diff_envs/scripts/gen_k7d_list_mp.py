from pb_diff_envs.environment.static.rand_kuka_env import RandKukaEnv
from pb_diff_envs.planner.sipp_planner import SippPlanner

import numpy as np
from importlib import reload
import h5py, argparse
import matplotlib.pyplot as plt
import pybullet as p
from pb_diff_envs.utils.kuka_utils_luo import SolutionInterp, visualize_kuka_traj_luo
import gym, pdb, traceback, os
from tqdm import tqdm
from pb_diff_envs.utils.utils import save_gif
import multiprocessing as mp
from multiprocessing import Process

from pb_diff_envs.utils.gen_kuka_utils_mp import gen_kuka_data_mp
from pb_diff_envs.utils.file_proc_utils import get_rest_idx
from pb_diff_envs.utils.gen_utils import check_data_len, npify
from pb_diff_envs.robot.grouping import RobotGroup
from pb_diff_envs.utils.robogroup_utils_luo import robogroup_visualize_traj_luo

def reset_data():
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

def append_data(data, s, a, tgt, done, wall_locations, maze_idx, reach_max_episode_steps, is_sol_kp, pos_reset):
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
    # checkparam not sure if the value is good
    interp_density = env_dvg_list.mazelist_config.get('interp_density', 3.0)
    args.vis_every = num_groups // 10 if num_groups >= 10 else 1 #  e.g. 500 -> 25: we vis 20 envs
    assert env_dvg_list.gen_num_parts == args.num_parts

    num_cores = args.num_cores
    # assert not hasattr(args, 'num_parts')
    # args.num_parts = args.num_cores
    num_parts = args.num_parts if args.num_parts is not None else num_cores
    args.num_parts = num_parts # for file proc
    # assert num_groups % args.num_parts == 0 or args.num_mazes > 0
    print("num_cores: " + str(num_cores))
    print("num_parts: " + str(num_parts))
    
    start = 0
    param_list = []
    # assert num_groups % num_cores == 0, f'num_groups:{num_groups}' # outdated delete
    each_num = num_groups // num_parts # num_cores, number of envs in each proc
    residual_num = num_groups % num_parts # num_cores # e.g. 2
    env_dvg_list.model_list = []
    # num_mp = num_parts + 1 if residual_num else num_parts #  num_cores + 1, num_cores
    num_mp = num_parts
    all_wall_locations = np.array(env_dvg_list.wallLoc_list, dtype=np.float32)[:,:,:3] # 15000, 20, 3

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
                          reset_data=reset_data,
                          append_data=append_data,)
        # copy.deepcopy(env_dvg_list) # very slow
        ## too young too simple, direcy passing would cause hanging
        # param = (pid, args, env_dvg_list, gen_config)
        param = (pid, args, None, gen_config) ## gpu2116

        param_list.append(param)
        start += each_num_tmp
    check_param_list(param_list, num_groups)
    if len(param_list) >= 2:
        print(param_list[0], param_list[1], param_list[-2], param_list[-1])
    # pdb.set_trace()

    ## -------- Test one Process --------
    # data = gen_kuka_data_mp(*param_list[0], lock=None)
    # npify(data)
    # check_data_len(data, len(data['observations']))
    # pdb.set_trace()
    ## ----------------------------------

    if args.gen_rest.strip() == "True":
        rest_dix = get_rest_idx(args, num_groups, './datasets/cache', args.env_name)
        param_list = [param_list[i_p] for i_p in rest_dix]
    # if args.reversed.strip() == "True":
        # param_list = param_list[::-1]
        # param_list = param_list[:250]
    # pdb.set_trace()
    print(f'len param_list: {len(param_list)}')

    
    # # print(f'env_dvg_list: {env_dvg_list} {id(env_dvg_list)}')
    # # print(f'env_dvg_list:', id(param_list[1][2]))
    # # with mp.Pool(processes=num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
    
    use_proc_pool = True
    if use_proc_pool:
        # Create a manager to provide shared resources
        manager = mp.Manager()
        lock = manager.Lock() # a shared lock, for Pool
        # assert mp.cpu_count() >= num_cores
        with mp.Pool(processes=num_cores) as pool:
            results_mp = []
            for i_mp in range(len(param_list)):
                result = pool.apply_async( gen_kuka_data_mp, args=param_list[i_mp]+(lock,) )
                results_mp.append(result)

            results_mp = [p.get() for p in results_mp]


    else:
        assert False, 'no need'
        

    print('multiprocess all finished.')

    # for i_r in range(len(results_mp)):
        # print(f'{i_r} {results_mp[i_r]}')

    data = aggregate_data_dict(results_mp)

    try:
        save_to_hdf5(args, data, env_dvg_list, num_groups, vis_dir)


    except Exception as e:
        traceback.print_exc()
        exit(0)


def aggregate_data_dict(data_list):
    data_agg = reset_data()
    for i in range(len(data_list)):
        data_tmp = data_list[i]
        for k,v in data_tmp.items():
            data_agg[k] += data_tmp[k]
    return data_agg

def save_to_hdf5(args, data, env_dvg_list, num_groups, vis_dir, vis_idx=None, is_ee3d=False):
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
        if isinstance(env_last.robot, RobotGroup):
            gifs, ds, vis_dict = robogroup_visualize_traj_luo(env_last, data['observations'][v_start:v_end],is_ee3d=is_ee3d)
        else:
            gifs, ds, vis_dict = visualize_kuka_traj_luo(env_last, data['observations'][v_start:v_end],is_ee3d=is_ee3d)
        save_gif(gifs, f'{vis_dir}/vis_{i_v}_last.gif', duration=ds)
    
    if args.gen_rest.strip() == "False":
        check_wallLoc_matched(data, env_dvg_list)

        with h5py.File(fname, 'w') as dataset:
            for k in data:
                dataset.create_dataset(k, data=data[k], compression='gzip')

    if 'testOnly' not in fname:
        os.chmod(fname, 0o444)
    wallLoc_list = np.array(env_dvg_list.wallLoc_list)

    npyname = '%s/%s.npy' % (prefix, args.env_name)
    np.save(npyname, wallLoc_list)

def check_wallLoc_matched(data, env_dvg_list):
    ## seems like also work in the case of dyn walls
    ## 500, 20, 3
    load_wallLoc = data['infos/wall_locations'][::env_dvg_list.samples_per_env]
    required_wallLoc =  np.array(env_dvg_list.wallLoc_list, dtype=np.float32)
    assert (load_wallLoc - required_wallLoc < 1e-4).all()

def check_param_list(param_list, num_groups):
    cnt = 0
    for param in param_list:
        cnt = cnt + param[-1]['end_idx'] - param[-1]['start_idx']
    assert cnt == num_groups, f'{cnt} {num_groups}'

if __name__ == '__main__':
    main()
