import numpy as np
from os.path import join
import diffuser.utils as utils
import torch, json, pdb, time, os
from collections import OrderedDict
from tqdm import tqdm
import pybullet as p
from pb_diff_envs.utils.kuka_utils_luo import SolutionInterp
from pb_diff_envs.utils.maze2d_utils import pad_traj2d_list
from diffuser.guides.rm2d_colchk import check_single_dyn_traj
import pandas as pd

def load_eval_problems_pb(env, pd_config):
    '''
    loading the evaluation problems for potential based diffusion
    '''

    if pd_config.get('load_unseen_maze', False):
        if pd_config.get('no_check_bit', False):
            npi = str(pd_config['npi'])
            # load no-checked problems
            problems_h5path = env._dataset_url.replace('.hdf5', f'-problems-nochk_{npi}pi.hdf5')
            assert False, "'bugfree, but don't do it now"
        else:
            assert pd_config.get('npi', 0) == 0
            # load checked problems
            problems_h5path = env._dataset_url.replace('.hdf5', '-problems.hdf5')

        problems_dict = env.get_dataset(h5path=problems_h5path, no_check=True) # a dict
    else:
        assert False

    return problems_dict




def kuka_obs_target_list(problems_dict, maze_idx, prob_permaze):
    '''
    ## ---------- Collect start and goal locations of a maze -------------
    return two list of numpy 
    env: MazeEnv
    prob_permaze: int have not multiple the repetitive factor here
    '''
    # extract
    # [n_models, n_problems, 2, 7] e.g. (500, 41, 2, 7)
    obs_start_list = problems_dict['problems'][maze_idx,:,0,:] # [n_models, n_problems, 7]
    target_list = problems_dict['problems'][maze_idx,:,1,:] # [n_models, n_problems, 7]

    assert prob_permaze <= obs_start_list.shape[0]
    # get partition
    obs_start_list = obs_start_list[:prob_permaze , :]
    target_list = target_list[:prob_permaze , :]

    return obs_start_list, target_list

# -------------------------------------------
# ------------- For replan ------------------
def check_start_end_repl(traj, st, gl, thres=0.4):
    assert type(traj) == type(st)
    return ( np.abs(traj[0] - st) < thres ).all() and ( np.abs(traj[-1] - gl) < thres ).all()

def check_start_end_repl_v2(ori_traj, new_section, c_list_i, checker, env):
    '''
    check if the replanned new section is 
    ori_traj 2d (h, d)
    new_section 2d (h, d)'''
    assert type(ori_traj) == type(new_section)
    assert ori_traj.ndim == 2 and new_section.ndim == 2
    valid = True
    cnt = 0
    # check 
    if c_list_i[0] > 0:
        v1, c1 = checker.steerTo_repl(  ori_traj[ c_list_i[0] - 1 ], new_section[0], env )
        valid &= v1
        cnt += c1

    if valid and ( c_list_i[-1] + 1 < len(ori_traj) ):
        v2, c2 = checker.steerTo_repl(  new_section[-1], ori_traj[ c_list_i[-1] + 1 ], env )
        valid &= v2
        cnt += c2
    
    return valid, cnt



def check_single_traj(env, traj):
    '''
    check collision of a single traj in a static env
    returns:
        collision cnt (int)
        a list of collision idx
    '''
    assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
    cnt, c_list = 0, []
    for i_t in range(len(traj)):
        
        env.robot.set_config(traj[i_t])            
        if not env.robot.no_collision():
            cnt += 1
            c_list.append(i_t)
        
    return cnt, c_list






def check_collision(env, trajs):
    '''
    check if collision exists throughout all points in the trajs
    args:
        env: the single planned env
        trajs: a list of trajectory in the single env, np (n_traj, horizon, 7/14)
    NOTE 
    returns:
        dict: a dict of multiple metrics
    '''
    assert trajs.ndim == 3
    collision_cnts = [] # how many collision in each traj
    success_list = []
    se_valid_list = []
    thres = trajs.shape[-1] ** 0.5 / 4
    for i_aj in range(len(trajs)): # tqdm
        traj = trajs[i_aj]

        se_valid = True # always true, because we set the start and goal manually
        
        if 'Dyn' in repr(env) and '2DEnv' in repr(env):
            # for dynamic 2d env
            wtrajs = env.problems_dict['infos/wall_locations'][i_aj] # 1d (96,)
            cnt, _ = check_single_dyn_traj(env, traj, wtrajs)
        else:
            # static env
            cnt, _ = check_single_traj(env, traj)

        collision_cnts.append(cnt)
        is_success = (cnt == 0 and se_valid)
        se_valid_list.append(se_valid)
        success_list.append(is_success)

    assert len(success_list) == len(trajs)
    traj_srate = np.array(success_list).sum() / len(success_list)
    se_valid_rate = np.array(se_valid_list).sum() / len(se_valid_list)
    no_collision_rate = (np.array(collision_cnts) == 0).sum() / len(collision_cnts)
    result_dict = dict(collision_cnts=collision_cnts, # not used now
                       success_list=success_list, # not used now
                       se_valid_rate=se_valid_rate, # start and end are good
                       traj_srate=traj_srate,
                       no_collision_rate=no_collision_rate)
    result_dict['num_samples'] = len(trajs)
    
    return result_dict


def plan_kuka_permaze_stat(vis_modelout_path_list, other_results, env):
    '''
    aggregate info and then append to a list
    '''
    trajs_len = []
    for i_aj, aj in enumerate(vis_modelout_path_list):
        trajs_len.append( str( (i_aj, len(aj)) ) )
    if vis_modelout_path_list[0].ndim == 3: # (1,h,d)
        trajs = np.concatenate(vis_modelout_path_list, axis=0)
    else:
        # pad for multi horizon
        vis_modelout_path_list = pad_traj2d_list( vis_modelout_path_list )
        trajs = np.stack(vis_modelout_path_list, axis=0)
    assert trajs.ndim == 3 # (B,h,d)

    # dict_keys(['collision_cnts', 'success_list', 'se_valid_rate', 'traj_srate', 'no_collision_rate', 'num_samples'])
    result_dict = check_collision(env, trajs) # per env result
    result_dict['trajs_len'] = trajs_len
    ori_len = len(result_dict)
    # dict_keys(['replan_suc_list', 'no_replan_suc_list', 'batch_runtime', 'num_colck_list'])
    result_dict.update(other_results)
    assert len(result_dict) == ori_len + len(other_results), 'no overlap'
    return result_dict


def compute_kuka_avg_result(result_dict_list):
    ''' a list of per env result dict
    the returned dict will be directly saved as global stats'''
    total_num = 0
    total_traj_s, total_svr, total_ncr = 0, 0, 0
    total_no_replan_s, total_replan_s = 0, 0
    total_b_time, total_num_colck = 0, 0
    ## env level loop through
    for d in result_dict_list:
        total_num += d['num_samples']
        total_traj_s += d['traj_srate'] * d['num_samples']
        total_ncr += d['no_collision_rate'] * d['num_samples']
        total_svr += d['se_valid_rate'] * d['num_samples']
        total_b_time +=  d['batch_runtime']
        total_no_replan_s += np.array(d['no_replan_suc_list']).sum()
        total_replan_s += np.array(d['replan_suc_list']).sum()
        total_num_colck += np.array( d.get('num_colck_list', float('inf')) ).sum().item()

    traj_srate = total_traj_s / total_num
    total_svr = total_svr / total_num
    total_ncr = total_ncr / total_num
    avg_batch_time = total_b_time / len(result_dict_list) # assume batch size is same
    avg_sample_time = total_b_time / total_num
    traj_norepl_srate = total_no_replan_s / total_num
    traj_repl_srate = total_replan_s / total_num
    avg_num_colck = total_num_colck / total_num
    result_dict = dict(num_samples=total_num,
                    traj_srate=traj_srate,
                    traj_norepl_srate=traj_norepl_srate,
                    traj_repl_srate=traj_repl_srate,
                    total_ncr=total_ncr,
                    total_svr=total_svr,
                    avg_batch_time=avg_batch_time,
                    avg_sample_time=avg_sample_time,
                    total_num_colck=total_num_colck,
                    avg_num_colck=avg_num_colck,
                    )

    return result_dict



def compute_sem(arr):
    arr = np.array( arr )
    if arr.ndim == 2:
        arr_mean = arr.mean( axis=1 )
    else:
        arr_mean = arr
    sem = pd.DataFrame( arr_mean ).sem().item()
    return sem



def compute_result_sem(result_dict_list):
    '''
    compute standard error
    '''
    sr, br, cc, ncl = [], [], [], []
    for i in range(len(result_dict_list)):
        d = result_dict_list[i]
        sr.append( d['success_list'] )
        ncl.append( d['num_colck_list'] )

        ## (total run time of all problems of one env) / (num of problems in one env)
        br.append( d['batch_runtime'] / len(d['success_list']) )
        cc.append( d['collision_cnts'] )
    d = dict()

    d['success_list_per_env_sem'] = compute_sem( sr )
    d['batch_runtime_per_env_sem'] = compute_sem( br )
    d['num_colck_list_per_env_sem'] = compute_sem(ncl)
    d['collision_cnts_per_env_sem'] = compute_sem( cc )

    return d


def plan_kuka_save_per_env_stat(result_dict_list, plan_env_config):
    # pdb.set_trace()
    json_data = OrderedDict()
    sem_tmp = compute_result_sem(result_dict_list)
    json_data.update(sem_tmp)

    for i in range(len(result_dict_list)):
        d = result_dict_list[i]
        key = f'env_{str(i)}:collision_cnts' # might be more for multi horizon
        json_data[ key ] = d['collision_cnts']
        key = f'env_{str(i)}:no_collision_rate'
        json_data[ key ] = d['no_collision_rate']
        key = f'env_{str(i)}:trajs_len'
        json_data[ key ] = d['trajs_len']

        key = f'env_{str(i)}:num_colck_list'
        json_data[ key ] = d['num_colck_list']
        key = f'env_{str(i)}:batch_runtime'
        json_data[ key ] = d['batch_runtime']
        key = f'env_{str(i)}:success_list'
        json_data[ key ] = bool( d['success_list'] )
    ## safe save
    for k, v in json_data.items():
        if isinstance(v, np.ndarray):
            json_data[k] = v.tolist()

    savepath = plan_env_config['savepath_dir']
    json_path = join(savepath, f'all-per_env-stat.json')
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=False)
    


def plan_pbdiff_save_all_stat(result_dict, maze_prefix, plan_env_config):
    '''result_dict is from compute_kuka_avg_result
    '''
    ## ------- save result as a json file ----------
    ## ---------------------------------------------

    savepath = plan_env_config['savepath_dir']
    str_epoch = plan_env_config['str_epoch']
    maze_idx = plan_env_config['maze_idx']

    json_path = join(savepath, f'all-{maze_prefix}{maze_idx}-rollout-{str_epoch}.json')

    json_data = OrderedDict([
        ('step', plan_env_config['n_timesteps']), # timestep of the diffusion model
        ('epoch_diffusion', plan_env_config['epoch']),
        # ('num_samples', result_dict['num_samples']), # total number of samples
        ('prob_permaze', plan_env_config['prob_permaze']),
        ('samples_perprob', plan_env_config['samples_perprob']),
        # ('no_collision_rate', result_dict['total_ncr']),
        # ('se_valid_rate', result_dict['total_svr']),
        # ('traj_success_rate', result_dict['traj_srate']),
        ('do_replan', plan_env_config['do_replan']),
        # ('avg_batch_time', result_dict['avg_batch_time']),
        # ('avg_sample_time', result_dict['avg_sample_time']),
        ('seed_maze', plan_env_config['seed_maze']),
        ('n_maze', plan_env_config['n_maze']),
        ('ddim', plan_env_config.get('use_ddim')),
        ])
    json_data.update(result_dict)
    for k, v in plan_env_config._dict.items():
        if 'js_' in k:
            json_data[k] = v

    tmp_src = json_path.split(os.sep)
    tmp_dst = json_path.split(os.sep)
    # tmp_dst[-2] = tmp_dst[-2] + f"-sr{result_dict['traj_srate']:.3f}"
    os.rename( src=os.sep.join(tmp_src[:-1]), dst=os.sep.join(tmp_dst[:-1]) )
    json_path = os.sep.join(tmp_dst)
    
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=False)
    print(f'[plan_pbdiff_save_all_stat]:', json_data)
    print(f'[plan_pbdiff_save_all_stat]: save to {json_path}')



def get_avg_dicts(list_of_dicts):
    '''return a dict that avg a list of dict'''
    averaged_dict = {}
    num_dicts = len(list_of_dicts)
    for dict_item in list_of_dicts:
        for key, value in dict_item.items():
            assert type(value) != list
            if key in averaged_dict:
                averaged_dict[key] += value
            else:
                averaged_dict[key] = value
    
    for key in averaged_dict:
        if key != 'num_samples':
            averaged_dict[key] /= num_dicts

    return averaged_dict



def plan_kuka_save_baseline_stat(baseline_dict_list, maze_prefix, plan_env_config):
    savepath = plan_env_config['savepath_dir']
    str_epoch = plan_env_config['str_epoch']
    maze_idx = plan_env_config['maze_idx']
    result_dict = get_avg_dicts(baseline_dict_list)
    utils.mkdir(savepath)

    json_path = join(savepath, f'all-{maze_prefix}{maze_idx}-rollout-baseline-{str_epoch}.json')

    json_data = OrderedDict([
        ('baseline', 'BIT'),
        ('num_samples', result_dict['num_samples']), # total number of samples
        ('prob_permaze', plan_env_config['prob_permaze']),
        # ('no_collision_rate', result_dict['total_ncr']),
        ('traj_success_rate', result_dict['traj_srate']),
        ('avg_suc_runtime', result_dict['avg_suc_runtime']),
        ('total_suc_runtime', result_dict['total_suc_runtime']),
        ('seed_maze', plan_env_config['seed_maze']),
        ('n_maze', plan_env_config['n_maze']),
        ])
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=False)
    print(f'[]:', json_data)
    print(f'[]: save to {json_path}')





def check_start_end(traj, thres=0.5): # should related to the density
    '''check if trajectory is close to the start and end'''
    s_valid = np.linalg.norm(traj[0] - traj[1]) < thres # 0.1, do use [1] - [0]
    e_valid = np.linalg.norm(traj[-2] - traj[-1]) < thres # 0.1
    cond_1 = s_valid & e_valid
    s_valid = np.linalg.norm(traj[2] - traj[1]) < thres # 0.1, do use [1] - [0]
    e_valid = np.linalg.norm(traj[-3] - traj[-2]) < thres # 0.1
    cond_2 = s_valid & e_valid
    return cond_1 & cond_2



def save_depoch_result(e_list, depoch_list, args):
    '''save avg results from a list of epoch, to check overfitting
      e_list: a list of return from avg, aka list of per epoch result  
      depoch_list: a list of int'''

    json_data = OrderedDict()
    savepath = args.savepath
    # find max srate (success rate)
    srates = []
    for i in range(len(e_list)):
        srates.append( e_list[i]['traj_srate'] )
    max_idx = np.array(srates).argmax()
    max_ep = depoch_list[ max_idx ]
    max_srate = srates[ max_idx ]
    json_data[ 'max' ] = dict(
        epoch=max_ep,
        traj_srate=max_srate)

    from glob import glob
    savepath_2 = glob( savepath.rstrip('/') + '**' )[0]


    json_path = join(savepath_2, f'depoch_summary_{max_ep//10000}_{srates[ max_idx ]}.json')

    for i in range(len(e_list)):
        json_data[  f'{depoch_list[i]}_{i}' ] = e_list[i]


    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=False)
    print(json_data)
    print(f'[save_depoch_result]: save to {json_path}')

