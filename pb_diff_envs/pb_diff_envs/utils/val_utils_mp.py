from pb_diff_envs.environment.static.rand_kuka_env import RandKukaEnv
import numpy as np
from importlib import reload
from colorama import Fore
from tqdm import tqdm

from .kuka_utils_luo import SolutionInterp, visualize_kuka_traj_luo
import time
from .utils import save_gif
import time
import pickle, gym, pdb, h5py, os
import multiprocessing as mp
import traceback
import logging
from .gen_kuka_utils_mp import save_sub_data
from pb_diff_envs.utils.gen_utils import check_data_len, npify
from pb_diff_envs.robot.grouping import RobotGroup
from .robogroup_utils_luo import robogroup_visualize_traj_luo

'''
Define Helper functions for generating problems using mutliprocessing,
actively developing
'''
def gen_val_kuka_data_mp(pid, args, env_dvg_list, gen_config, lock, result_queue=None): # shared_lock
    '''
    cousin but should be different from the gen_kuka_data_mp
    This is for validation problems, the other for training traj.
    '''
    reset_data = gen_config['reset_data']
    append_data = gen_config['append_data']

    start_idx = gen_config['start_idx']
    end_idx = gen_config['end_idx'] # not included

    vis_dir = gen_config['vis_dir']
    interp_density = gen_config['interp_density']
    envs_wallLoc = gen_config['envs_wallLoc'] # 100, 20, 7 ?
    no_check_bit = gen_config.get('no_check_bit', False)

    data = reset_data()
    sol_interp = SolutionInterp(density=interp_density)
    print(f'env_dvg_list: {env_dvg_list} {id(env_dvg_list)}') # different across process
    print(f'args: {id(args)}') # different across process


    ## check if dead lock!
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"), multiproc_mode=True, is_eval=True, debug_mode=True) # checkparam eval mode
    env_dvg_list.proc_id = pid
    if args.npi != 0:
        ml_cfg = {'min_episode_distance': args.npi * np.pi}
    else:
        ml_cfg = {}
    env_dvg_list.mazelist_config.update( ml_cfg )

    '''start generating data in each env'''
    for maze_idx in range(start_idx, end_idx):
        print(Fore.RED + f'maze_idx {maze_idx} started.' + Fore.RESET)
        # 1. load urdf to get robot info
        # print(f"{pid} Process (maze_idx {maze_idx}) is trying to acquire the lock. {id(lock)}")
        # print(f"{pid} Process has acquired the lock.")
        while True:
            try:
                env = env_dvg_list.create_single_env(maze_idx) # create env here
                print(Fore.RED + f'maze_idx {maze_idx} min_episode_distance {env.min_episode_distance}.' + Fore.RESET)
                ## unload old env, reduce connection.
                if maze_idx > start_idx:
                    env_dvg_list.model_list[maze_idx-1].unload_env()
            except Exception as e:
                traceback.print_exc()
                logging.error(traceback.format_exc())
                print(Fore.GREEN + f'pid {pid} maze_idx {maze_idx} create_single_env error' + Fore.RESET)
                time.sleep(5)
                continue
            else:
                break

        # print(f"{pid} Process has *release* the lock.")
        assert (env.wall_locations - envs_wallLoc[maze_idx-start_idx] < 1e-3).all(), f'maze_idx: {maze_idx}'
        if env_dvg_list.train_eval_no_overlap:
            ## check no overlap
            pass

        # 2. load urdf to get robot/obj/table
        while True:
            try:
                env.load(GUI=False)
            except Exception as e:
                traceback.print_exc()
                # logging.error(traceback.format_exc())
                print(Fore.GREEN + f'pid {pid} maze_idx {maze_idx}' + Fore.RESET)
                # might be useful, hang sometime when pb_id = 1 due to my assert
                # if p.isConnected():
                    # p.disconnect()
                time.sleep(5)
                continue
            else:
                break
        # print('collision_eps', env.robot.collision_eps)



        ## 3. For loop to get enough problems
        start = np.array([0.0] * env.robot.config_dim)
        env_problems = []
        env_pl_time = []
        env_n_colchk = []
        assert args.num_samples_permaze <= 500
        for i_p in range(args.num_samples_permaze):
            env.robot.set_config(start)
            if gen_config['rng_val'] is None:
                rng_val = np.random.default_rng(seed=None)
            else:
                assert False
                rng_val = gen_config['rng_val']

            if no_check_bit:
                start_end, pl_time, n_colchk = env.sample_1_val_episode_no_check() # [2, 7]
            else:
                ## normal checked solution
                start_end, pl_time, n_colchk = env.sample_1_val_episode(start, rng_val) # [2, 7]
            env_problems.append(start_end)
            env_pl_time.append(pl_time)
            env_n_colchk.append(n_colchk)

            # print('ori start', start)
            start = np.copy(start_end[-1])

            # assert env.state_fp(start, 0)
            assert env.state_fp(start)


        # 4. This env is finished, reformat data and append
        env_problems = np.expand_dims(np.stack(env_problems, axis=0), axis=0) # [1, n, 2, 7]

        pose_diff = np.abs(env_problems[:, :, 1, :] - env_problems[:,:, 0,:]) # [1, n, 7]
        # assert ( np.sum( pose_diff, axis=-1 ) > 3 ).all(), 'every problem should be non-trivial' # [1, n]


        wallLoc = env.o_env.get_obstacles()[..., :3] # (20,6) [Bug] only wall locations 

        wall_locations = np.tile(wallLoc, (args.num_samples_permaze, 1, 1))
        # (1, n_probs, num_walls, 3): (1, 100 ,20, 3)
        wall_locations = np.expand_dims(wall_locations, axis=0)
        env_maze_idx = np.full(shape=(1,), fill_value=maze_idx, dtype=np.int32)
        env_maze_idx = np.expand_dims(env_maze_idx, axis=0) # (1,1)

        env_pl_time = np.expand_dims( np.array(env_pl_time), axis=0 ) # (1, n_p)
        env_n_colchk = np.expand_dims( np.array(env_n_colchk), axis=0 )

        ## data of one env
        append_data(data, env_problems, wall_locations, env_maze_idx, env_pl_time, env_n_colchk)



        if maze_idx % args.vis_every == 0: # True:

            ## do some visualization to check
            vis_traj = env_problems[0, 0:5, ...].copy() # [5, 2, 7]
            vis_traj = vis_traj.reshape( -1, env.robot.config_dim ) # [10, 7]
            # env_traj = vis_traj
            env_traj = sol_interp(vis_traj)
            while True:
                try:
                    if isinstance(env.robot, RobotGroup):
                        gifs, ds, vis_dict = robogroup_visualize_traj_luo(env, env_traj, lock)
                    else:
                        gifs, ds, vis_dict = visualize_kuka_traj_luo(env, env_traj, lock)
                except Exception as e:
                    traceback.print_exc()
                    print(Fore.BLUE + f'pid {pid} visualize_kuka_traj_luo error' + Fore.RESET)
                    time.sleep(2)
                    continue
                else:
                    break
            # gifs, ds, vis_dict = visualize_kuka_traj_luo(env, env_traj[:150], lock)
            gifs_fname = f'{vis_dir}/vis_{maze_idx}.gif'
            save_gif(gifs, gifs_fname, duration=ds)

    
    ## 6. --- this part is finished ---
    pid_str = f'eval_{pid}' # to separate from the training subdata
    if no_check_bit: 
        pid_str = f'eval_no_check_{pid}'
    save_sub_data(data, pid_str, args)
    print(f'pid {pid} finished')
    env.unload_env()
    
    ## used when not using mp.Pool but use Process()
    if result_queue is not None:
        try:
            print(Fore.GREEN + f'pid {pid} result_queue 1' + Fore.RESET)
            result_queue.put((pid, data), block=True)
            print(f'pid {pid} result_queue 2')
        except Exception as e:
            traceback.print_exc()
    return data


def val_check_wallLoc_matched(data, env_dvg_list):
    ## ng, np, n_wall, 3
    load_wallLoc = data['infos/wall_locations']
    ## ng, n_wall, 3
    required_wallLoc =  np.array(env_dvg_list.wallLoc_list, dtype=np.float32)
    ## ng, 1, n_wall, 3
    required_wallLoc = np.expand_dims(required_wallLoc, axis=1)
    assert (load_wallLoc - required_wallLoc < 1e-4).all()

def get_val_prob_fname(args):
    prefix = './datasets'
    fname = '%s/%s-problems.hdf5' % (prefix, args.env_name)
    if getattr(args,'no_check_bit', False):
        fname = '%s/%s-problems-nochk_%spi.hdf5' % (prefix, args.env_name, str(args.npi))
    return fname

def is_val_prob_exist(args):
    return os.path.isfile(get_val_prob_fname(args))

def val_save_to_hdf5(args, data, env_dvg_list, num_groups, vis_dir, vis_idx=None, is_ee3d=False):
    '''save the whole dataset'''
    fname = get_val_prob_fname(args)
    if not type(data['problems']) == np.ndarray:
        npify(data)
    if not getattr(args,'no_check_bit', False):
        check_data_len(data, env_dvg_list.eval_num_groups) # len(data['problems'])
    val_check_wallLoc_matched(data, env_dvg_list)

    
    with h5py.File(fname, 'w') as dataset:
        for k in data:
            dataset.create_dataset(k, data=data[k], compression='gzip')
    
    if 'testOnly' not in fname:
        ## prevent overwrite
        os.chmod(fname, 0o444)
    