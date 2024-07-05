import numpy as np
from colorama import Fore
from tqdm import tqdm
import time
from pb_diff_envs.utils.utils import save_gif
from .gen_kuka_utils_mp import save_sub_data
import time
import pickle, gym
import multiprocessing as mp
import traceback
import logging
import pdb
from pb_diff_envs.environment.static.maze2d_rand_wgrp import Maze2DRandRecGroupList
from pb_diff_envs.environment.static.rand_maze2d_env import RandMaze2DEnv
from .kuka_utils_luo import SolutionInterp
from .maze2d_utils import split_cont_traj

def gen_dot_sm2d_data_mp(pid, args, env_dvg_list, gen_config, lock, result_queue=None):
    reset_data = gen_config['reset_data']
    append_data = gen_config['append_data']

    start_idx = gen_config['start_idx']
    end_idx = gen_config['end_idx'] # not included

    vis_dir = gen_config['vis_dir']
    interp_density = gen_config['interp_density']
    envs_wallLoc = gen_config['envs_wallLoc'] # 100, 20, 7
    copy_times = gen_config['copy_times']

    data = reset_data()
    if True:
        sol_interp = SolutionInterp(density=interp_density)
    print(f'env_dvg_list: {env_dvg_list} {id(env_dvg_list)}') # different across process
    print(f'args: {id(args)}') # different across process

    ## check if dead lock!
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"), multiproc_mode=True)
    # pdb.set_trace()

    # assert type(env_dvg_list.env.env.env) == Maze2DRandRecGroupList
    env_dvg_list.proc_id = pid

    '''start generating data in each env'''
    for maze_idx in range(start_idx, end_idx):
        print(f"{pid} Process (maze_idx {maze_idx}) is trying to acquire the lock. {id(lock)}")
        print(f"{pid} Process has acquired the lock.")
        while True:
            try:
                env = env_dvg_list.create_single_env(maze_idx) # create env here
                # assert type(env) == RandMaze2DEnv, 'is different in dynamic'
                ## ** Not Unload might be the reason that cause hanging **
                ## unload old env, reduce connection.
                if maze_idx > start_idx:
                    env_dvg_list.model_list[maze_idx-1].unload_env()
            except Exception as e:
                traceback.print_exc()
                logging.error(traceback.format_exc())
                print(Fore.GREEN + f'pid {pid} create_single_env error' + Fore.RESET)
                time.sleep(5)
                continue
            else:
                break
        
        print(f"{pid} Process has *release* the lock.")

        # pdb.set_trace()
        assert (env.wall_locations -  envs_wallLoc[maze_idx-start_idx] < 1e-3).all(), f'maze_idx: {maze_idx}'

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

        ## ** Different from Kuka **
        # start = np.array([0.0] * env.robot.config_dim) # len(env.robot.joints)
        # start = env.get_robot_free_pose() # start from a random valid pose

        ## append the start
        env_traj = [] # will be np
        env_done = []
        env_is_sol_kp = []
        env_pos_reset = []
        len_env_traj = 0 # the start
        env_goal = []
        env_no_col = []


        while len_env_traj < args.num_samples_permaze:
            # env.robot.set_config(start)
            ## the start/end may be duplicate
            ## solution: [start, p1, p2, ..., goal]

            if False:
                while True:
                    pose = np.random.uniform(low=env.robot.limits_low, high=env.robot.limits_high)
                    env.robot.set_config(pose)
                    if env.state_fp(pose):
                        solution = pose[None,].repeat( copy_times, axis=0 ) # h, 7
                        break
            else:
                pose = np.random.uniform(low=env.robot.limits_low, high=env.robot.limits_high)
                solution = pose[None,] # (1, 7) .repeat( copy_times, axis=0 ) # h, 7
                env.robot.set_config(pose)
                no_col = np.array([env.robot.no_collision(),]) # make it a np1d (1,)
            

            # solution, sol_dict = env.sample_1_episode(start)


            ## sol_dict['pos_reset'] is a scalar bool 
            # start = np.copy(solution[-1])

            ## bool array with len = solution
            is_sol_keypoint = np.zeros(shape=(copy_times,),dtype=bool )

            goal_len = len(solution)
            goal = np.tile(pose, (goal_len, 1)) # (84, 7)

            len_env_traj += len(solution)
            
            # print('solution', solution)
            env_traj.append(solution) # 1
            done = np.zeros(shape=(len(solution,)), dtype=bool)
            # done[-1] = True
            env_done.append(done) # 2
            env_goal.append(goal) # 3
            env_is_sol_kp.append(is_sol_keypoint)
            env_no_col.append( no_col )

            pos_reset = np.zeros(shape=(len(solution,)), dtype=bool)
            env_pos_reset.append(pos_reset)

            

        ### --------- post process the generated data -----------
        ## trim the traj to ...
        env_traj = np.concatenate(env_traj)
        env_done = np.concatenate(env_done)
        env_goal = np.concatenate(env_goal)
        env_pos_reset = np.concatenate(env_pos_reset)
        env_is_sol_kp = np.concatenate(env_is_sol_kp)
        env_no_col = np.concatenate( env_no_col ) # 1d (B,)

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
        env_no_col = env_no_col[:args.num_samples_permaze]


        # --------    Staic     --------
        # assert type(env) == RandMaze2DEnv, 'is different in dynamic'
        wallLoc = np.copy(env.wall_locations)
        # pdb.set_trace()
        assert wallLoc.shape[1] == env_dvg_list.world_dim and wallLoc.ndim == 2
        ## (x, y, z, hExt, hExt, hExt)
        wall_locations = np.tile(wallLoc, (args.num_samples_permaze, 1, 1)) # (500, 6, 2)
        # wall_locations = wall_locations[..., :3] # (500, 20, 6 -> 3) remove cube size
        # assert wall_locations.shape == (env_dvg_list.samples_per_env, env_dvg_list.num_walls, env_dvg_list.world_dim)
        assert (wall_locations -  envs_wallLoc[None, maze_idx-start_idx] < 1e-3).all(), f'maze_idx: {maze_idx}'
        # -----------------------------

        print(Fore.RED + f'maze_idx {maze_idx} checked & finished.' + Fore.RESET)


        env_maze_idx = np.full(shape=env_done.shape, fill_value=maze_idx, dtype=np.int32)
        reach_max_episode_steps = np.zeros_like(env_done, dtype=bool)


        ## args: (data, s, a, tgt, done, wall_locations, maze_idx, reach_max_episode_steps)
        append_data(data, env_traj, act, env_goal, env_done, wall_locations, \
                        env_maze_idx, reach_max_episode_steps, env_is_sol_kp, env_pos_reset, env_no_col)


        assert ( np.sum( np.abs(env_traj[1:] - env_traj[:-1]), axis=1) <= 1e-3 ).sum() <= 10


        
        ## ------------ visualization -------------

        # if maze_idx % args.vis_every == 0: # True:

        #     tmp = f'{vis_dir}/vis_{maze_idx}_c.png'
            
        #     trajs = split_cont_traj(env_traj, env_done,)
        #     env.render_composite(tmp, trajs)
        #     tmp = f'{vis_dir}/vis_{maze_idx}.png'
        #     # inside data: a list of np2d
        #     env.render_1_traj(tmp, env_traj[:500])


    save_sub_data(data, pid, args)
    print(f'pid {pid} finished')
    env.unload_env()
    

    return data
