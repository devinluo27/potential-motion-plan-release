import time, sys, h5py
sys.path.append('.')
import numpy as np
import argparse
import gym, pdb, traceback, os
import multiprocessing as mp
from pb_diff_envs.utils.val_utils_mp import val_save_to_hdf5, is_val_prob_exist
from pb_diff_envs.utils.run_trad_planner_utils import eval_m2d_trad_planner_mp, dyn_rm2d_val_probs_preproc
from pb_diff_envs.scripts.gen_k7d_list_mp import check_param_list
from pb_diff_envs.environment.static.maze2d_rand_wgrp import Maze2DRandRecGroupList
from tqdm import tqdm
from pb_diff_envs.utils.gen_utils import check_data_len, npify
from pb_diff_envs.utils.file_proc_utils import get_notexist_savepath
from pb_diff_envs.scripts.gen_static_m2d_val_probs_mp import reset_data_m2d, append_data_m2d # ,val_aggregate_data_dict_m2d
from pb_diff_envs.planner.bit_star_planner import BITStarPlanner
from pb_diff_envs.planner.rrt_star_planner import RRTStarPlanner
from pb_diff_envs.planner.sipp_planner import SippPlanner

from pb_diff_envs.utils.diffuser_utils import filter_json_serializable

def reset_data_m2d_trad():
    tmp = reset_data_m2d()
    tmp['eval_sols'] = []
    tmp['is_suc'] = []
    return tmp

def append_data_m2d_trad(data, prob, wall_locations, maze_idx, pl_time, n_colchk, eval_sols, is_suc):
    append_data_m2d(data, prob, wall_locations, maze_idx, pl_time, n_colchk,)
    data['eval_sols'].append( eval_sols.copy() )
    data['is_suc'].append( is_suc )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default=None, help='kuka7d env id')
    # TODO change default here
    parser.add_argument('--num_samples_permaze', type=int, help='Num problems to collect')
    parser.add_argument('--num_mazes', type=int, default=-1, help='Num of mazes (override env if > 0). Used when create evaluation placeholder data.')
    parser.add_argument('--load_mazeEnv', type=str, default="True", help='')
    parser.add_argument('--num_cores', type=int, default=3, help='')
    parser.add_argument('--num_parts', type=int, default=None, help='') # 150?
    parser.add_argument('--planner', type=str, choices=['bit', 'rrt', 'p-rrt', 'sipp', 'rmp']) # 150?
    parser.add_argument('--prob_suffix', type=str, default='') # 150?
    parser.add_argument('--timeout', type=int, default=None) # 150?
    args = parser.parse_args()
    
    vis_dir = f'./visualization/{args.env_name}_eval_{args.planner}/'
    os.makedirs(name=vis_dir, exist_ok=True)
    
    env_dvg_list = gym.make(args.env_name, load_mazeEnv=(args.load_mazeEnv.strip() != "False"),\
                            is_eval=True)
    
    env_dvg_list: Maze2DRandRecGroupList
    ## eval groups
    num_groups = args.num_mazes # num_groups if args.num_mazes <= 0 else args.num_mazes
    # assert args.num_mazes == 100 # env_dvg_list.eval_num_groups
    # assert args.num_samples_permaze == 20 # env_dvg_list.eval_num_probs

    # checkparam not sure if the value is good
    interp_density = env_dvg_list.mazelist_config['interp_density']
    args.vis_every = num_groups // 30 if num_groups >= 30 else 1 #  e.g. 200 -> 20: we vis 10 envs

    num_cores = args.num_cores
    num_parts = args.num_parts
    assert num_groups % args.num_parts == 0 # or num_groups < args.num_parts
    print("num_cores: " + str(num_cores))
    print("num_parts: " + str(num_parts))
    from utils.utils import print_color
    print_color( f'Tested on {num_groups} envs, {args.num_samples_permaze}' )


    ## 1. ---- load problems to be evaluted ----
    problems_h5path = env_dvg_list.dataset_url.replace('.hdf5', f'-problems{args.prob_suffix[1:]}.hdf5')
    problems_dict = env_dvg_list.get_dataset(h5path=problems_h5path, no_check=True)

    ## ------ load planner --------
    if args.planner == 'rrt':
        planner_val = RRTStarPlanner(stop_when_success=True)
    elif args.planner == 'bit':
        planner_val = BITStarPlanner(num_batch=400, stop_when_success=True)
    elif args.planner == 'sipp':
        planner_val = SippPlanner(num_samples=1000,)
        dyn_rm2d_val_probs_preproc(problems_dict, (0,1), env_dvg_list)
    else:
        raise NotImplementedError

    
    probs =  problems_dict['problems']
    

    
    ## 2. ------ create list of params ------
    start = 0
    param_list = []
    each_num = num_groups // num_parts # num_cores, number of envs in each proc
    residual_num = num_groups % num_parts # num_cores # e.g. 2
    assert residual_num == 0 #
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
                          reset_data=reset_data_m2d_trad,
                          append_data=append_data_m2d_trad,
                          probs=probs,
                          planner_val=planner_val,
                          infos_wlocs=problems_dict['infos/wall_locations'],
                          timeout=args.timeout,
                          )

        param = (pid, args, None, gen_config) ## args will be copy automatically
        param_list.append(param)
        start += each_num_tmp

    check_param_list(param_list, num_groups)
    # print(param_list[0], param_list[1], param_list[-2], param_list[-1])

    ## -------- Test one Process --------
    # data = eval_m2d_trad_planner_mp(*param_list[0], lock=None)
    # npify(data)
    # check_data_len(data, len(data['problems']))
    # ## save_trad_result('./tmptmptmp.hdf5', data, filter_json_serializable(planner_val.__dict__) )
    # pdb.set_trace()
    ## ----------------------------------



    ## 3. ------ Create Process ------
    # Create a manager to provide shared resources
    manager = mp.Manager()
    lock = manager.Lock() # a shared lock, for Pool


    # assert mp.cpu_count() >= num_cores
    with mp.Pool(processes=num_cores) as pool:
        results_mp = []
        for i_mp in range(len(param_list)):
            result = pool.apply_async( eval_m2d_trad_planner_mp, args=param_list[i_mp]+(lock,) )
            results_mp.append(result)
        results_mp = [p.get() for p in results_mp]



    data = val_aggregate_data_dict_m2d_trad(results_mp)

    print(f"final eval_sols: {len(data['eval_sols'])}, {len(data['eval_sols'][0])},"
           f"{data['eval_sols'][0][0].shape}")
    assert len(data['eval_sols']) == args.num_mazes
    assert len(data['eval_sols'][0]) == args.num_samples_permaze
    

    prefix = './logs/trad_planner/'
    fname = '%s/%s-eval-%s.hdf5' % (prefix, args.env_name, args.planner)

    save_trad_result(fname, data, filter_json_serializable(planner_val.__dict__))
    print('run traditional planner val problems finished.')



def save_trad_result(fname, data, attr_info=None):
    npify(data)
    fname = get_notexist_savepath(fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    with h5py.File(fname, 'w') as dataset:
        for k in data:
            tmp = len(data[k]) >= 20
            if k == 'eval_sols':
                pass
            else:
                dataset.create_dataset(k, data=data[k], compression='gzip')
        if attr_info:
            for k in attr_info:

                attr_k = attr_info[k] if attr_info[k] is not None else 'None'
                dataset.attrs[k] = attr_k
    
    import pickle
    fname_pkl = fname.replace('.hdf5', '.pkl')
    fname_pkl = get_notexist_savepath(fname_pkl)
    with open(fname_pkl, 'wb') as f:
        pickle.dump(data['eval_sols'], f)
    
    if 'testOnly' not in fname and tmp:
        os.chmod(fname, 0o444)
        os.chmod(fname_pkl, 0o444)
    print(f'save to {fname}')
    print(f'save sols to {fname_pkl}')
    print( 'suc rate:', np.array(data['is_suc']).mean() )


def val_aggregate_data_dict_m2d_trad(data_list):
    data_agg = reset_data_m2d_trad()
    for i in range(len(data_list)):
        data_tmp = data_list[i]
        for k,v in data_tmp.items():
            assert type(data_tmp[k]) == list, f'key: {k}'
            data_agg[k] += data_tmp[k]
    return data_agg



if __name__ == '__main__':
    for i in range(1):
        main()


