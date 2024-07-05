from diffuser.guides.kuka_plan_utils import check_single_traj
import numpy as np
import pybullet as p
from os.path import join
import pdb
import diffuser.utils as utils
import pickle
import os, contextlib


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as nullfile:
        with contextlib.redirect_stdout(nullfile):
            yield



def get_all_suc_trajs(env, trajs, goal):
    '''given a list of trajs, return the success ones [in a list],
    used in picking static kuka trajs.
    trajs: list of np2d, unnormed
    goal: 1D e.g., (7,)
    '''
    env.load(GUI=False)
    suc_trajs, suc_idxs = [], []
    for i, traj in enumerate(trajs):
        # cnt, c_list = check_single_traj( env, np.array(traj) )

        no_col = True
        for i_t in range(len(traj)-1):
            no_col = env.edge_fp(traj[i_t], traj[i_t+1]) and no_col
            if not no_col:
                break

        reach_g = np.linalg.norm(traj[-1] - goal) < 0.2
        if no_col and reach_g:
            suc_trajs.append(traj.copy())
            suc_idxs.append(i)
    env.unload_env()
    print(f'suc_idxs: {suc_idxs}')
    return suc_trajs, suc_idxs

        


def kuka_save_vis_gif_4paper(renderer, maze_prefix, vis_modelout_path_list, \
    vis_maze_idxs_list, plan_env_config, suc_list_env):
    '''vis failure cases only'''

    prob_permaze = plan_env_config['prob_permaze'] # 80
    samples_perprob = plan_env_config['samples_perprob'] # 2
    # wall_locations_list = plan_env_config['wall_locations_list'] # None if single maze
    maze_idx = plan_env_config['maze_idx']
    savepath = plan_env_config['savepath_dir']

    num_vis_traj = plan_env_config['num_vis_traj'] # visualize the first num_vis samples
    vis_start_idx = plan_env_config.get('vis_start_idx', 2)
    # problems_dict = plan_env_config['problems_dict']
    # wall_locations = problems_dict['infos/wall_locations']

    total_samples = prob_permaze * samples_perprob
    num_vis = num_vis_traj # * samples_perprob # 1


    assert len(vis_maze_idxs_list) == total_samples and len(vis_modelout_path_list) == total_samples

    assert len(np.unique(vis_maze_idxs_list)) == 1, 'should be the same'
    assert type(vis_modelout_path_list[0]) == np.ndarray
    # assert vis_modelout_path_list[0].ndim == 3 and len(vis_modelout_path_list[0]) == 1


    str_epoch = plan_env_config['str_epoch']

    ## save direct prediction
    vis_end_idx = vis_start_idx + num_vis * samples_perprob
    if vis_modelout_path_list[0].ndim == 2: # list of 2d(horizon, 7)
        pass
    else:
        ## we only visualize a portion from list
        ## ensure render path in shape = [ n_paths x horizon x 2 ]
        # vis_modelout_path_list = np.concatenate(vis_modelout_path_list[:vis_end_idx], axis=0)
        fail_list_env = np.logical_not( np.array(suc_list_env) )
        pdb.set_trace()
        vis_modelout_path_list = np.concatenate(vis_modelout_path_list[ fail_list_env ], axis=0)


    fail_list_env = np.where( np.array(suc_list_env) == False, )[0]
    path_dict = { i: vis_modelout_path_list[i].copy() for i in fail_list_env }
    vis_modelout_path_list = [ vis_modelout_path_list[i] for i in fail_list_env ]
    tmp_str = ''
    for xx in fail_list_env.tolist():
        tmp_str += f'_{str(xx)}'
    fullpath = join(savepath, f'{maze_prefix}{maze_idx}-modelpred-{tmp_str}-{str_epoch}.png')
    vis_maze_idxs_list = vis_maze_idxs_list[ 0: len(vis_modelout_path_list) ] # same, simply copy

    pkl_path = fullpath.replace('.png', '.pkl')

    tmp_dict = dict(vis_modelout_path_list=vis_modelout_path_list, 
                path_dict=path_dict,
                vis_maze_idxs_list=vis_maze_idxs_list )
    with open( pkl_path, 'wb' ) as f_pkl:
        pickle.dump(tmp_dict, f_pkl)
        print(f'[vis4paper]: save pickle traj: {pkl_path}')

    
    renderer.composite(fullpath, vis_modelout_path_list, maze_idx=vis_maze_idxs_list)


def wloc_center_mode_to_3w(wloc, hExt, mode):
    ## hExt is the large one
    wlocs = []; divd = 1
    wlocs.append( [wloc[0] - hExt[0] / divd, wloc[1] + hExt[1] / divd] )
    wlocs.append( wloc + hExt / divd )
    wlocs.append( [wloc[0] + hExt[0] / divd, wloc[1] - hExt[1] / divd] )
    wlocs.append( wloc - hExt / divd )
    wlocs.pop(  round(mode-1) )
    # print(round(mode-1))
    wlocs = np.array(wlocs)
    # print(wlocs)
    return wlocs

def create_every_wloc_43(wlocs, hExt):
    # wlocs: (N, 2+1), mode
    # hExt shape=(2,)
    wlocs_new = []
    for i_w, wloc in enumerate(wlocs):
        wlocs_new.append(  wloc_center_mode_to_3w(wloc[:2], hExt, wloc[2]) )
    wlocs_new = np.concatenate( wlocs_new, axis=0 )
    return wlocs_new