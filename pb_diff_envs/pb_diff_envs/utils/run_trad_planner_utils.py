import numpy as np
from colorama import Fore
import time
from utils.utils import save_gif
import pickle, gym, pdb, h5py, os, einops
import multiprocessing as mp
import traceback
import logging
from pb_diff_envs.utils.gen_kuka_utils_mp import save_sub_data
import pdb
from pb_diff_envs.environment.static.rand_maze2d_env import RandMaze2DEnv
from pb_diff_envs.environment.dynamic_env import DynamicEnv
from pb_diff_envs.environment.abs_maze2d_env import AbsDynamicMaze2DEnv
from pb_diff_envs.utils.kuka_utils_luo import SolutionInterp
from pb_diff_envs.utils.maze2d_utils import pad_traj2d_list
from pb_diff_envs.utils.utils import print_color

def eval_m2d_trad_planner_mp(pid, args, env_dvg_list, gen_config, lock, result_queue=None): # shared_lock
    '''
    This is runing RRT, BIT on validation problems
    '''
    reset_data = gen_config['reset_data']
    append_data = gen_config['append_data']

    start_idx = gen_config['start_idx']
    end_idx = gen_config['end_idx'] # not included

    vis_dir = gen_config['vis_dir']
    interp_density = gen_config['interp_density']
    envs_wallLoc = gen_config['envs_wallLoc'] # 100, 20, 7
    probs = gen_config['probs']
    planner_val = gen_config['planner_val']
    infos_wlocs = gen_config['infos_wlocs']


    data = reset_data()
    sol_interp = SolutionInterp(density=interp_density)
    print(f'env_dvg_list: {env_dvg_list} {id(env_dvg_list)}') # different across process
    print(f'args: {id(args)}') # different across process

    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"), multiproc_mode=True, is_eval=True, debug_mode=False) # checkparam eval mode
    # assert type(env_dvg_list.env.env.env) in [Maze2DRandRecGroupList, ComposedRM2DGroupList]
    print_color(f'env_dvg_list: {env_dvg_list.env}', c='c')

    env_dvg_list.proc_id = pid

    '''start generating data in each env'''
    for maze_idx in range(start_idx, end_idx):
        print(Fore.RED + f'maze_idx {maze_idx} started.' + Fore.RESET)
        # 1. load urdf to get robot info
        while True:
            try:
                env = env_dvg_list.create_single_env(maze_idx) # create env here
                # assert type(env) in  [RandMaze2DEnv, ComposedRM2DEnv], 'is different in dynamic'
                print_color(f'env: {env}', c='c')
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
        if not isinstance(env, AbsDynamicMaze2DEnv):
            assert (env.wall_locations - envs_wallLoc[maze_idx-start_idx] < 1e-3).all(), f'maze_idx: {maze_idx}'

        # 2. load urdf to get robot/obj/table
        env.load(GUI=False)
        # print('collision_eps', env.robot.collision_eps)

        ## 3. For loop to get enough problems
        env_is_suc = []
        env_solutions = []
        env_problems = []
        env_pl_time = []
        env_n_colchk = []
        assert args.num_samples_permaze <= 20, 'num of problems'
        for i_p in range(args.num_samples_permaze):
            # env.robot.set_config(start)
            if gen_config['rng_val'] is None:
                rng_val = np.random.default_rng(seed=None)
            else:
                assert False

            # start_end, pl_time, n_colchk = env.sample_1_val_episode(start, rng_val) # [2, 7]
            if isinstance(env, AbsDynamicMaze2DEnv):
                # pdb.set_trace()
                # infos_wlocs: n_env, h, nw, 2
                wtrajs = infos_wlocs[maze_idx, i_p] # 
                env.set_dyn_wallgroup(wtrajs)
                env.wall_locations = wtrajs


            # (n_env, n_p, 2, dim)
            prev_pos = probs[maze_idx, i_p, 0] # (dim, )
            new_goal = probs[maze_idx, i_p, 1] # (dim, )
            start_end = probs[maze_idx, i_p].copy() # (2, dim)
            env.robot.set_config(prev_pos)
            timeout = gen_config.get('timeout') if gen_config.get('timeout') else env.planner_timeout 
            result_tmp = planner_val.plan(env, prev_pos, new_goal, timeout=('time', timeout))


            sol = result_tmp.solution
            if sol is None:
                sol = np.zeros(shape=(2, env.robot.config_dim), dtype=np.float32)
                is_suc = False
            else:
                sol = np.array(sol)
                ## append start and goal pose to two ends
                # pdb.set_trace()
                # prev_pos, new_goal (2,)
                sol = np.concatenate( [sol, new_goal[None,]], axis=0 )
                
                is_suc = check_1_traj_trad(env, sol, prev_pos, new_goal)
            
            pl_time = result_tmp.running_time
            n_colchk = result_tmp.num_collision_check

            env_solutions.append(sol) # a list of [np2d]
            env_is_suc.append(is_suc)
            env_problems.append(start_end)
            env_pl_time.append(pl_time)
            env_n_colchk.append(n_colchk)

            # start = np.copy(start_end[-1])

            # assert env.state_fp(start)

        # 4. This env is finished, reformat data and append
        env_problems = np.expand_dims(np.stack(env_problems, axis=0), axis=0) # [1, n, 2, 2]
        
       
        # --------------
        ## env_solutions is a list of np2d
        env_is_suc = np.expand_dims( np.array(env_is_suc, dtype=bool), axis=0 )
        
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
        env_n_colchk = np.expand_dims( np.array(env_n_colchk) ,axis=0 )
        # -----------------------------

        ## data of one env
        append_data(data, env_problems, wall_locations, env_maze_idx, env_pl_time, env_n_colchk, env_solutions, env_is_suc)

        print('is suc', env_is_suc)
        print('pl time', env_pl_time)



        if maze_idx % args.vis_every == 0: # True:
            

            if isinstance(env, AbsDynamicMaze2DEnv):
                pass
            elif 'static_maze2d' in getattr(env, 'env_id', ''):
                rm2d_vis_trad(env, env_solutions, vis_dir, maze_idx)
            else:
                kuka_vis_trad(env, env_solutions, vis_dir, maze_idx, sol_interp)


    
    ## 6. --- this part is finished ---
    pid_str = f'eval_{pid}' # to separate from the training subdata
    # save_sub_data(data, pid_str, args) # no need
    print(f'pid {pid} finished')
    env.unload_env()
    
    ## used when not using mp.Pool but use Process()
    if result_queue is not None:
        assert False
    return data




def check_1_traj_trad(env, r_traj, st, gl):
    assert type(r_traj) == np.ndarray
    tmp1 = np.abs(st - r_traj[0]).sum() < 1e-1
    tmp2 = np.abs(gl - r_traj[-1]).sum() < 1e-1
    if not (tmp1 and tmp2):
        return False
    ## assume static
    for i in range( len(r_traj)-1 ):
        if isinstance(env, DynamicEnv): # actually a static env
            no_col = env.edge_fp(r_traj[i], r_traj[i+1], 0, 0)
        elif isinstance(env, AbsDynamicMaze2DEnv):
            no_col = env.edge_fp(r_traj[i], r_traj[i+1], i, i+1)
            # raise NotImplementedError()
        else:
            no_col = env.edge_fp(r_traj[i], r_traj[i+1])
        if not no_col:
            return False
    return True

from typing import List
from robot.grouping import RobotGroup
from utils.robogroup_utils_luo import robogroup_visualize_traj_luo
from utils.kuka_utils_luo import SolutionInterp, visualize_kuka_traj_luo

def rm2d_vis_trad(env: RandMaze2DEnv, 
                  env_solutions: List[np.ndarray], vis_dir, maze_idx):
    vis_trajs = []
    for i_v in range( min(20, len(env_solutions)) ):
        # traj = sol_interp(env_solutions[i_v]) # [2, 2] -> [30, 2]
        traj = env_solutions[i_v]
        vis_trajs.append( traj )

    tmp = f'{vis_dir}/vis_{maze_idx}_c_.png'

    env.render_composite(tmp, vis_trajs)
    tmp = f'{vis_dir}/vis_{maze_idx}.png'
    # inside data: a list of np2d
    env.render_1_traj(tmp, np.concatenate(vis_trajs, axis=0))


def kuka_vis_trad(env, 
                  env_solutions: List[np.ndarray], vis_dir, maze_idx, sol_interp):
    # pdb.set_trace()
    n_vis = min( len(env_solutions), 5 )
    vis_traj = np.concatenate( env_solutions[:5], axis=0 )
    if len(vis_traj) < n_vis * 10:
        vis_traj = sol_interp(vis_traj)
    
    while True:
        try:
            if isinstance(env.robot, RobotGroup):
                gifs, ds, vis_dict = robogroup_visualize_traj_luo(env, vis_traj[:300], lock=None)
            else:
                gifs, ds, vis_dict = visualize_kuka_traj_luo(env, vis_traj[:150], lock=None)
        except Exception as e:
            traceback.print_exc()
            print(Fore.BLUE + f'maze_idx {maze_idx} visualize_kuka_traj_luo error' + Fore.RESET)
            time.sleep(2)
            continue
        else:
            break
    
    gifs_fname = f'{vis_dir}/vis_{maze_idx}.gif'
    save_gif(gifs, gifs_fname, duration=ds)



def dyn_rm2d_val_probs_preproc(dataset, wloc_select, env):
    '''copy from diffuser'''
    assert env.is_eval
    infos_wloc = dataset['infos/wall_locations']
    probs = dataset['problems']
    assert infos_wloc.ndim == 4
    
    '''
    preprocessing the val problems to [n_envs, n_prob_env, 2, dim]
    simply extract the start and end every 48 horizon
    when training, the input to wloc is
    flatten: w1_t1, w2_t1, w1_t2, w2_t2, ...,
    '''

    # numpy: n_p, horizon, n_w, 2
    infos_wloc = infos_wloc[..., wloc_select]
    total_p, horizon, n_w, w_dim = infos_wloc.shape
    
    # we select a divisible number of problems
    n_env = 100
    n_prob_per_env = total_p // n_env
    select_prob = n_env * n_prob_per_env
    infos_wloc = infos_wloc[:select_prob, ...] # 4000, 96
    # reshape 
    tmp = infos_wloc.reshape(n_env, n_prob_per_env, horizon, n_w, w_dim)
    # infos_wloc = einops.rearrange(infos_wloc, '(n_e n_ppe) d -> n_e n_ppe d', n_e=n_env, n_ppe=n_prob_per_env)
    infos_wloc = tmp.transpose( (0, 1, 3, 2, 4) )
    # assert ( tmp == infos_wloc).all()
    
    # should be [n_envs, n_prob_env, -1]
    dataset['infos/wall_locations'] = infos_wloc



    probs = probs[:select_prob, ...] # 4000, 48, dim
    probs = einops.rearrange( probs, '(n_e n_ppe) h d -> n_e n_ppe h d', n_e=n_env, n_ppe=n_prob_per_env)
    # (100, 40, 48, 2) -> (100, 40, 2, 2), extract start/goal
    probs = probs[:, :, (0,-1), :]
    # should be [n_envs, n_prob_env, 2, dim]
    dataset['problems'] = probs


