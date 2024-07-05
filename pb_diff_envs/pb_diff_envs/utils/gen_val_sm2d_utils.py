import numpy as np
from colorama import Fore
import time
from .utils import save_gif
import pickle, gym, pdb, h5py, os
import multiprocessing as mp
import traceback
import logging
from .gen_kuka_utils_mp import save_sub_data
import pdb
from pb_diff_envs.environment.static.maze2d_rand_wgrp import Maze2DRandRecGroupList
from pb_diff_envs.environment.static.maze2d_rand_wgrp_43 import Maze2DRandRecGroupList_43
from pb_diff_envs.environment.comp.comp_rand_m2d_wgrp import ComposedRM2DGroupList
from pb_diff_envs.environment.static.rand_maze2d_env import RandMaze2DEnv
from pb_diff_envs.environment.static.comp_rm2d_env import ComposedRM2DEnv
from .kuka_utils_luo import SolutionInterp

def gen_val_m2d_data_mp(pid, args, env_dvg_list, gen_config, lock, result_queue=None): # shared_lock
    '''
    This is for validation problems, the other for training traj.
    '''
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

    ## check if dead lock!
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"), multiproc_mode=True, is_eval=True, debug_mode=False) # checkparam eval mode
    assert type(env_dvg_list.env) in [Maze2DRandRecGroupList, ComposedRM2DGroupList, Maze2DRandRecGroupList_43]
    env_dvg_list.proc_id = pid

    '''start generating data in each env'''
    for maze_idx in range(start_idx, end_idx):
        print(Fore.RED + f'maze_idx {maze_idx} started.' + Fore.RESET)
        # 1. load env
        while True:
            try:
                env = env_dvg_list.create_single_env(maze_idx) # create env here
                assert type(env) in [RandMaze2DEnv, ComposedRM2DEnv,], 'is different in dynamic' 

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
        
        assert (env.wall_locations - envs_wallLoc[maze_idx-start_idx] < 1e-3).all(), f'maze_idx: {maze_idx}'

        # 2. load urdf to get robot/obj/table
        env.load(GUI=False)
        # print('collision_eps', env.robot.collision_eps)



        ## 3. For loop to get enough problems
        ## ** Different from Kuka **
        # start = np.array([0.0] * env.robot.config_dim) # len(env.robot.joints)
        start = env.get_robot_free_pose() # start from a random valid pose
        env_problems = []
        env_pl_time = []
        env_n_colchk = []
        assert args.num_samples_permaze <= 500, 'num of problems'
        for i_p in range(args.num_samples_permaze):
            env.robot.set_config(start)
            if gen_config['rng_val'] is None:
                rng_val = np.random.default_rng(seed=None)
            else:
                assert False

            start_end, pl_time, n_colchk = env.sample_1_val_episode(start, rng_val) # [2, 7]
            env_problems.append(start_end)
            env_pl_time.append(pl_time)
            env_n_colchk.append(n_colchk)

            start = np.copy(start_end[-1])

            assert env.state_fp(start)

        # 4. This env is finished, reformat data and append
        env_problems = np.expand_dims(np.stack(env_problems, axis=0), axis=0) # [1, n, 2, 2]

        pose_diff = np.abs(env_problems[:, :, 1, :] - env_problems[:,:, 0,:]) # [1, n, 2]
        # assert ( np.sum( pose_diff, axis=-1 ) > 2.5 ).all(), 'every problem should be non-trivial' # [1, n]

        # --------    Staic     --------

        # wallLoc = env.o_env.get_obstacles()[..., :env_dvg_list.world_dim] # (20,2) [Bug] only wall locations 
        wallLoc = np.copy(env.wall_locations)
        wall_locations = np.tile(wallLoc, (args.num_samples_permaze, 1, 1))
        # (1, n_probs, num_walls, 3): (1, 100 ,20, 3)
        wall_locations = np.expand_dims(wall_locations, axis=0)
        env_maze_idx = np.full(shape=(1,), fill_value=maze_idx, dtype=np.int32)
        env_maze_idx = np.expand_dims(env_maze_idx, axis=0) # (1,1)
        env_pl_time = np.expand_dims( np.array(env_pl_time), axis=0 ) # (1, n_p)
        env_n_colchk = np.expand_dims( np.array(env_n_colchk), axis=0 )
        # -----------------------------

        ## data of one env
        append_data(data, env_problems, wall_locations, env_maze_idx, env_pl_time, env_n_colchk)



        if maze_idx % args.vis_every == 0: # True:
            ## do some visualization to check
            vis_probs = env_problems[0, 0:5, ...].copy() # [5, 2, 2]
            vis_trajs = []
            for i_v in range(len(vis_probs)):
                traj = sol_interp(vis_probs[i_v]) # [2, 2] -> [30, 2]
                vis_trajs.append( traj )

            tmp = f'{vis_dir}/vis_{maze_idx}_c_.png'

            env.render_composite(tmp, vis_trajs)
            tmp = f'{vis_dir}/vis_{maze_idx}.png'
            # inside data: a list of np2d
            env.render_1_traj(tmp, np.concatenate(vis_trajs, axis=0))


    
    ## 6. --- this part is finished ---
    pid_str = f'eval_{pid}' # to separate from the training subdata
    save_sub_data(data, pid_str, args)
    print(f'pid {pid} finished')
    env.unload_env()
    
    ## used when not using mp.Pool but use Process()
    if result_queue is not None:
        assert False
    return data