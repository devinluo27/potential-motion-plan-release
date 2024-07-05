import numpy as np
from importlib import reload
from colorama import Fore
from tqdm import tqdm
from .kuka_utils_luo import SolutionInterp, visualize_kuka_traj_luo
import time
from .utils import save_gif
import time
import pickle, gym
import multiprocessing as mp
import traceback
import logging, os
from .file_proc_utils import get_files_with_prefix, get_number_from_fname
import pdb
from pb_diff_envs.robot.grouping import RobotGroup
from pb_diff_envs.utils.robogroup_utils_luo import robogroup_visualize_traj_luo
'''
Define Helper functions for generating problems using mutliprocessing,
'''

def gen_kuka_data_mp(pid, args, env_dvg_list, gen_config, lock, result_queue=None): # shared_lock
    reset_data = gen_config['reset_data']
    append_data = gen_config['append_data']

    start_idx = gen_config['start_idx']
    end_idx = gen_config['end_idx'] # not included

    vis_dir = gen_config['vis_dir']
    interp_density = gen_config['interp_density']
    envs_wallLoc = gen_config['envs_wallLoc'] # 100, 20, 7

    data = reset_data()
    sol_interp = SolutionInterp(density=interp_density)
    print(f'env_dvg_list: {env_dvg_list} {id(env_dvg_list)}') # different across process
    print(f'args: {id(args)}') # different across process

    ## solve dead lock issue
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"), multiproc_mode=True)
    env_dvg_list.proc_id = pid

    '''start generating data in each env'''
    for maze_idx in range(start_idx, end_idx):
        
        print(f"{pid} Process (maze_idx {maze_idx}) is trying to acquire the lock. {id(lock)}")
        print(f"{pid} Process has acquired the lock.")

        while True:
            try:
                env = env_dvg_list.create_single_env(maze_idx) # create env here
                ## ** Not Unload might be the reason that cause hanging **
                ## unload old env, reduce connection.
                if maze_idx > start_idx:
                    env_dvg_list.model_list[maze_idx-1].unload_env()
            except Exception as e:
                traceback.print_exc()
                logging.error(traceback.format_exc())
                # lock.release()
                print(Fore.GREEN + f'pid {pid} create_single_env error' + Fore.RESET)
                time.sleep(5)
                # lock.acquire()
                continue
            else:
                break

        print(f"{pid} Process has *release* the lock.")

        assert (env.wall_locations -  envs_wallLoc[maze_idx-start_idx] < 1e-7).all(), f'maze_idx: {maze_idx}'

        while True:
            try:
                env.load(GUI=False)
            except Exception as e:
                traceback.print_exc()
                logging.error(traceback.format_exc())
                time.sleep(5)
                continue
            else:
                break

        start = np.array([0.0] * env.robot.config_dim) # len(env.robot.joints)

        ## append the start
        env_traj = [start.reshape(1, -1),] # will be np
        env_done = [np.zeros(shape=(1,),dtype=bool), ]
        env_is_sol_kp = [np.array([1,], dtype=bool), ]
        env_pos_reset = [np.zeros(shape=(1,),dtype=bool),]
        len_env_traj = 1 # the start
        env_goal = []

        while len_env_traj < args.num_samples_permaze:
            env.robot.set_config(start)
            ## the start/end may be duplicate
            ## solution: [start, p1, p2, ..., goal]
            while True:
                try:
                    solution, sol_dict = env.sample_1_episode(start)
                except Exception as e:
                    traceback.print_exc()
                    time.sleep(5)
                    print(Fore.BLUE + f'pid {pid} sample_1_episode error' + Fore.RESET)
                    continue
                else:
                    break
            ## sol_dict['pos_reset'] is a scalar bool 
            start = np.copy(solution[-1])

            solution, infos = sol_interp(solution, True)
            ## bool array with len = solution
            is_sol_keypoint = infos['is_sol_keypoint'][1:] ## exclude idx 0 (duplicate)

            # if sol_dict['pos_reset']: # a bool to be deleted
                # e.g. [2,7] -> [3,7]
                # solution = np.concatenate([solution[None, 0],  solution], axis=0)

            goal_len = len(solution) if len(env_goal) == 0 else len(solution) - 1
            goal = np.tile(start, (goal_len, 1)) # (84, 7)

            # print('sol_interp solution', solution)
            # print('new start solution[:-1]', solution[:-1])
            
            # start should have been added
            solution = solution[1:] # not include start
            len_env_traj += len(solution)
            
            # print('solution', solution)
            env_traj.append(solution) # 1
            done = np.zeros(shape=(len(solution,)), dtype=bool)
            done[-1] = True
            env_done.append(done) # 2
            env_goal.append(goal) # 3
            env_is_sol_kp.append(is_sol_keypoint)

            pos_reset = np.zeros(shape=(len(solution,)), dtype=bool)
            pos_reset[0] = sol_dict['pos_reset']
            env_pos_reset.append(pos_reset)


            # assert env.state_fp(start, 0)
            assert env.state_fp(start)

        ### --------- post process the generated data -----------
        ## trim the traj to ...
        env_traj = np.concatenate(env_traj)
        env_done = np.concatenate(env_done)
        env_goal = np.concatenate(env_goal)
        env_pos_reset = np.concatenate(env_pos_reset)
        env_is_sol_kp = np.concatenate(env_is_sol_kp)
        # last_goal = env_goal[-1]
        # env_goal[env_goal == last_goal] = env_goal[args.num_samples_permaze-1].reshape(1, -1)

        # pdb.set_trace()
        # print('env_traj', env_traj.shape) # e.g. (564, 7)

        env_traj = env_traj[:args.num_samples_permaze]
        # no action is needed, shape:(500, 1)
        act = np.zeros(shape=(env_traj.shape[0], 1), dtype=np.float32)
        env_goal = env_goal[:args.num_samples_permaze]
        env_done = env_done[:args.num_samples_permaze]
        env_done[-1] = True
        env_is_sol_kp = env_is_sol_kp[:args.num_samples_permaze]
        env_pos_reset = env_pos_reset[:args.num_samples_permaze]

        wallLoc = env.o_env.get_obstacles()
        ## (x, y, z, hExt, hExt, hExt)
        wall_locations = np.tile(wallLoc, (args.num_samples_permaze, 1, 1)) # (500, 20, 6)
        wall_locations = wall_locations[..., :3] # (500, 20, 6 -> 3) remove cube size


        assert (wall_locations -  envs_wallLoc[None, maze_idx-start_idx] < 1e-3).all(), f'maze_idx: {maze_idx}'
        print(Fore.RED + f'maze_idx {maze_idx} checked & finished.' + Fore.RESET)


        env_maze_idx = np.full(shape=env_done.shape, fill_value=maze_idx, dtype=np.int32)
        reach_max_episode_steps = np.zeros_like(env_done, dtype=bool)


        ## args: (data, s, a, tgt, done, wall_locations, maze_idx, reach_max_episode_steps)
        append_data(data, env_traj, act, env_goal, env_done, wall_locations, \
                        env_maze_idx, reach_max_episode_steps, env_is_sol_kp, env_pos_reset)


        assert ( np.sum( np.abs(env_traj[1:] - env_traj[:-1]), axis=1) > 1e-2 ).all()

        if maze_idx % args.vis_every == 0: # True:
            ## do some visualization to check
            while True:
                try:
                    if isinstance(env.robot, RobotGroup):
                        gifs, ds, vis_dict = robogroup_visualize_traj_luo(env, env_traj[:300], lock)
                    else:
                        gifs, ds, vis_dict = visualize_kuka_traj_luo(env, env_traj[:150], lock)
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

    save_sub_data(data, pid, args)
    print(f'pid {pid} finished')
    env.unload_env() # this is important, because the process will be used...
    
    ## used when not using mp.Pool but use Process()
    if result_queue is not None:
        try:
            print(Fore.GREEN + f'pid {pid} result_queue 1' + Fore.RESET)
            result_queue.put((pid, data), block=True)
            print(f'pid {pid} result_queue 2')
        except Exception as e:
            traceback.print_exc()
    return data

def save_sub_data(data, pid, args):
    '''each process save data separately'''
    prefix = './datasets/cache/'
    os.makedirs(prefix, exist_ok=True)
    fname = f'{prefix}/{args.env_name}_{pid}.pkl' # noisy is added in env id
    # create a binary pickle file 
    with open(fname, "wb") as f_pkl:
        pickle.dump(data, f_pkl)
    print(Fore.RED + f'[save_sub_data] pid {pid}: {fname}' + Fore.RESET)


def load_all_sub_data(args, env_dvg_list):
    folder_path = './datasets/cache/'
    pkl_list = get_files_with_prefix(folder_path, prefix=args.env_name)
    pkl_list = sorted(pkl_list, key=get_number_from_fname) ## must sort

    assert ( len(pkl_list) * args.each_num ) == env_dvg_list.num_groups
    results_mp = []
    for i in tqdm(range(len(pkl_list))):
        fname = pkl_list[i]
        with open(fname, "rb") as f_pkl:
            ## data_i is a dict
            ## each value in data_i is a list (e.g.len 100) of numpy
            data_i = pickle.load(f_pkl)
            data_len = env_dvg_list.samples_per_env * args.each_num
            # a list of 100, each elem is np (500, 7)
            assert len(data_i['observations']) * data_i['observations'][0].shape[0]  == data_len

            results_mp.append(data_i)

    return results_mp
