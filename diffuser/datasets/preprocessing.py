import gym
import numpy as np
import einops
import pdb
import pickle

from .data_api import load_environment

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn

def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)

def get_policy_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)

#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#



def rm2d_kuka_preproc(dataset, wloc_select=(0,1,2), env=None):
    ''' 
    A general preprocessing functions for both maze2d and kuka/dual kuka
    NOTE: perform in-placed modification
    wall_locations might be of dimension 6 for kuka, where (:3) xyz, (3:) size; mostly we only use the xyz
    '''
    assert len(wloc_select) == 3 or 'kuka' not in env.name.lower()
    
    info_wlocs = dataset['infos/wall_locations']
    if info_wlocs.ndim == 2 and info_wlocs.shape[-1] > 10: # already flatted?
        return
    
    dataset['infos/wall_locations'] = dataset['infos/wall_locations'][..., wloc_select]
    if dataset['infos/wall_locations'].ndim == 3: # (B, n_cubes, 3)
        ## randomize wall location order in MLP
        if env is not None:
            assert not env.is_eval, 'must in train mode'
            
        ## just flatten
        ds_len = len(dataset['infos/wall_locations'])
        dataset['infos/wall_locations'] = dataset['infos/wall_locations'].reshape(ds_len, -1)
    elif dataset['infos/wall_locations'].ndim == 4: # (n_envs, n_probs, n_cubes, 3)

        # here, static rm2d or kuka
        n_envs, n_prob_env =  dataset['infos/wall_locations'].shape[:2]
        # n_env, n_p, n_w*dim
        dataset['infos/wall_locations'] = dataset['infos/wall_locations'].reshape(n_envs, n_prob_env, -1)
        

    else:
        ## Dynamic env
        raise NotImplementedError()
    


def dyn_rm2d_val_probs_preproc(dataset, wloc_select, env):
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

    infos_wloc = infos_wloc.reshape(total_p, horizon, n_w * w_dim)
    infos_wloc = infos_wloc.reshape(total_p, -1)
    
    # we select a divisible number of problems
    n_env = 100
    n_prob_per_env = total_p // n_env
    select_prob = n_env * n_prob_per_env
    infos_wloc = infos_wloc[:select_prob, ...] # 4000, 96
    # reshape 
    tmp = infos_wloc.reshape(n_env, n_prob_per_env ,-1)
    infos_wloc = einops.rearrange(infos_wloc, '(n_e n_ppe) d -> n_e n_ppe d', n_e=n_env, n_ppe=n_prob_per_env)
    assert ( tmp == infos_wloc).all()
    
    # should be [n_envs, n_prob_env, -1]
    dataset['infos/wall_locations'] = infos_wloc


    probs = probs[:select_prob, ...] # 4000, 48, dim
    probs = einops.rearrange( probs, '(n_e n_ppe) h d -> n_e n_ppe h d', n_e=n_env, n_ppe=n_prob_per_env)
    # (100, 40, 48, 2) -> (100, 40, 2, 2), extract start/goal
    probs = probs[:, :, (0,-1), :]
    # should be [n_envs, n_prob_env, 2, dim]
    dataset['problems'] = probs



"""
[an update version] **only for our generated paths**
we keep the original episode len, 
e.g.,for mazelist_small min length: 1 | max length: 307,

This function is also for diffusion kuka 7d
"""
def maze2d_set_terminals_luo(env):
    env_name = env
    env = load_environment(env) if type(env) == str else env

    def _fn(dataset):
        """
        A function to segment a long sequence into paths
        Just use the original terminals sigal to segment the paths!
        """
        ## TODO: Just for kuka
        if 'kuka' in env.name.lower():
            rm2d_kuka_preproc(dataset, env=env)
        elif 'MultiW-randS' in env.name: # add the hExt to the 3rd dim
            rm2d_kuka_preproc(dataset, wloc_select=(0,1,2), env=env)
        elif 'randS' in env.name:
            rm2d_kuka_preproc(dataset, wloc_select=(0,1), env=env)
        assert dataset['infos/wall_locations'].ndim == 2
        

        ## 1. prevent different maze_idx in one episode
        ## set terminals = true if it is the last sample of a maze
        ## NOTE this will results in some very small episode length 1,2
        if 'maze_idx' in dataset.keys():
            print('[preprocessing] prevent different maze_idx in one episode')
            maze_idx =  dataset['maze_idx'].copy()
            # False means maze change; last sample is set to False
            maze_idx_tmp = np.append(maze_idx[:-1] == maze_idx[1:], False)
            maze_change_idx = np.where(maze_idx_tmp == False,)[0]
            # print('maze_change_idx', maze_change_idx)
            dataset['terminals'][maze_change_idx] = True
            # Added on June 9: True is maze change at the point
            dataset['maze_change_bool'] = np.zeros_like(dataset['terminals'], dtype=bool)
            dataset['maze_change_bool'][maze_change_idx] = True

            

        timeouts = dataset['terminals'].copy()
        
        ## ---- Just for Printing Stat, Begin-------
        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        # the first path is missing, +1 due to idx starts from 0
        path_lengths = np.insert(path_lengths, 0, timeout_steps[0]+1, axis=0)
        print('[ utils/preprocessing ] path_lengths', path_lengths[:10])
        print(f"[ utils/preprocessing ] dataset['observations'] {dataset['observations'].shape}", 'path_lengths', path_lengths.shape)

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
            f', path here == step bewteen timeouts'
        )
        ## ---- Just for Printing Stat, End -------

        dataset['timeouts'] = timeouts
        return dataset

    return _fn






def get_pklname_43(env_list_name, is_eval):
    prefix = "pb_diff_envs/pb_diff_envs/datasets/rand_rec2dgrp/"
    if is_eval:
        npyname = f'{prefix}/{env_list_name}_43_eval.pkl'
    else:
        npyname = f'{prefix}/{env_list_name}_43.pkl'
    return npyname



def append_center_mode_43(env):
    file_path = get_pklname_43(env.name, env.is_eval)
    def _fn(dataset):
        with open(file_path, 'rb') as file:
            data43 = pickle.load(file)
            cpos = data43['centers_43'] # (12, 7, 2)
            mode = data43['mode_list'][..., None] # (12, 7) -> (12, 7, 1)
        
        if 'maze_change_bool' not in dataset.keys():
            compute_maze_change_bool(dataset)
        
        aa, bb = np.where(dataset['maze_change_bool'])[0][0:2]
        rep = bb - aa # repeat time

        new_wloc = np.concatenate( [cpos, mode], axis=-1 ) # n_envs, nw, 3

        new_wloc = new_wloc.reshape(cpos.shape[0], -1)[:, None, :]  # n_envs, 1, nw*3
        new_wloc = np.repeat(new_wloc, rep, axis=1) # n_envs, nsample_perenv , nw*3
        dataset['infos/wall_locations'] = einops.rearrange(new_wloc, 'ne nsp d -> (ne nsp) d')


        return dataset
    return _fn



def preproc_val_srm2dwloc_43(env, n_probs):
    # 1. load center and mode; 2. return (n_env, n_probs, nw, dim)
    import pickle
    file_path = get_pklname_43(env.name, True)
    with open(file_path, 'rb') as file:
        data43 = pickle.load(file)
        cpos = data43['centers_43'] # (300, 7, 2)
        mode = data43['mode_list'][..., None] # -> (300, 7, 1)

    new_wloc = np.concatenate( [cpos, mode], axis=-1 ) # n_envs, nw, 3 == (300, 7, 3)
    ## required (300, 200, nw, dim)
    # n_probs = self.problems_dict['infos/wall_locations'].shape[1]
    new_wloc = new_wloc[:, None, :, :].repeat( n_probs , axis=1 )
    # self.problems_dict['infos/wall_locations'] = new_wloc
    return new_wloc
    

def compute_maze_change_bool(dataset):
    maze_idx = dataset['maze_idx'].copy()
    # False means maze change; last sample is set to False
    maze_idx_tmp = np.append(maze_idx[:-1] == maze_idx[1:], False)
    maze_change_idx = np.where(maze_idx_tmp == False,)[0]
    # print('maze_change_idx', maze_change_idx)
    dataset['terminals'][maze_change_idx] = True
    # True if maze change at the point
    dataset['maze_change_bool'] = np.zeros_like(dataset['terminals'], dtype=bool)
    dataset['maze_change_bool'][maze_change_idx] = True







def shuffle_along_axis(sh_seed, arr, wall_dim, axis=1):
    '''
    Args:
    sh_seed: seed to shuffle
    arr: numpy B, n_walls, 3
    shuffle the arr along dim 1
    Returns:
    np (B, n_walls, 3)
    '''
    assert arr.ndim == 3 and axis == 1
    rng = np.random.default_rng(seed=sh_seed)

    idx = rng.random(  size=arr.shape[:(axis+1)] ) # B, N,
    idx = idx.argsort(axis=axis)
    idx = idx[..., None]
    idx = np.tile(idx, (1, 1, wall_dim))
    return np.take_along_axis(arr, idx, axis=axis)