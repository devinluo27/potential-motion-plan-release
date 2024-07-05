from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from pb_diff_envs.environment.dynamic_rrgroup import DynamicRectangleWall, DynamicRecWallGroup
from pb_diff_envs.objects.trajectory import WaypointLinearTrajectory
from pb_diff_envs.environment.rand_rec_group import RectangleWall, RectangleWallGroup
from typing import List
import imageio, os

class SplineInterp:
    def __init__(self, k=1) -> None:
        self.k = k # degree, 1: linear

    def fit_bspline(self, xy: np.array, n_pts_after: int, figsize, is_plot):
        assert xy.ndim == 2 and xy.shape[-1] == 2, 'xy should be (n, 2)'
        assert n_pts_after >= len(xy) * 3, 'otherwise, maynot cross given points'
        x = np.copy(xy[:, 0])
        y = np.copy(xy[:, 1])
        # shape must be like [2, np]
        tck, u = interpolate.splprep( [x, y] , k=self.k, s=0, per=False)
        u_new = np.linspace(0, 1, n_pts_after) # must be between 0, 1
        # list of np, (n_D, n_pts)
        new_points = interpolate.splev(u_new, tck)
        if is_plot:
            # print('new_points', new_points)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(0 - 1, figsize[0] + 1)
            ax.set_ylim(0 - 1, figsize[1] + 1)
            # ax.set_ylim(5.4, figsize[1] + 1)
            ax.scatter(x, y, color='b', label='given')
            ax.scatter(new_points[0], new_points[1], color='r', marker='x', label='interp')
            ax.plot(new_points[0], new_points[1],)
            ax.legend(loc='upper right', ncol=1)
            plt.grid(True)        
            plt.show()
            plt.clf()
            plt.close()
        new_points = np.array(new_points).T # (n_pts, 2)
        return new_points
    
    def fit_line_same_stepsize(self, pt_1, pt_2, speed: float, is_plot=False):
        '''
        Input two dots,
        assume that distance between pt1 and pt2 are *divisible* by speed
        sipp needs to take a traj with same speed.
        speed: gap between two waypoints
        '''
        assert pt_1.shape == (2,) and pt_2.shape == (2,)
        # print('point2 1', point2)
        # point2 = round_down_to_nearest(point2, speed)
        # print('point2 2', point2)
        diff_vector = pt_2 - pt_1 # should be a multiple of speed
        # pt_2_round = round_to_nearest(pt_2, speed)
        # assert ( np.abs(pt_2_round - pt_2) < 1e-2 ).all(), 'otherwise, might cause collision'
        diff_length = np.linalg.norm(diff_vector)
        diff_length_round = round_to_nearest(diff_length, speed)
        assert np.abs(diff_length_round - diff_length) < 1e-2 , 'otherwise, might cause collision'

        n_new_pts = np.round(diff_length / speed).astype(int) # floor
        
        # +1: [0, 0.25, 0.5, 0.75, 1]
        ts = np.linspace(0, 1, n_new_pts+1).reshape(-1, 1)
        new_pts = pt_1.reshape(1, -1) + ts * diff_vector.reshape(1, -1) # 2d(n, 2), same step
        print(f'fit line new_pts: {new_pts.shape}, len: {diff_length}')

        if is_plot:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(0 - 1, 6 + 1)
            ax.set_ylim(0 - 1, 6 + 1)
            ax.scatter(pt_1[0], pt_1[1], color='b', marker='x', s=50)
            ax.scatter(pt_2[0], pt_2[1], color='r', marker='x', s=50)
            ax.scatter(new_pts[:,0], new_pts[:,1], color='b', s=2)
            ax.grid(True)
            plt.close(fig)
        return new_pts











def get_is_collision_wtrajs(r_traj, wtrajs, hExts, min_to_wall_dist=0.01) -> np.ndarray:
    """used in visualization"""
    
    assert r_traj.shape[0] == wtrajs.shape[1]
    recWall_list = []

    for i_w in range(len(wtrajs)):
        wpl_traj = WaypointLinearTrajectory(wtrajs[i_w])
        tmp = DynamicRectangleWall(wtrajs[i_w, 0], 
                                    hExts[i_w], wpl_traj)
        recWall_list.append(tmp)

    tmp_grp = DynamicRecWallGroup(recWall_list)
    is_cols = []
    for i_r in range(len(r_traj)):
        tmp_grp.set_walls_at_time(i_r)
        is_col = tmp_grp.is_point_inside_wg(r_traj[i_r], min_to_wall_dist)
        is_cols.append(is_col)
    return np.array(is_cols)


def get_is_collision_static(r_traj, center_pos_list, hExt_list, min_to_wall_dist=0.01) -> np.ndarray:
    """used in visualization"""
    
    recWall_list = []

    for i_w in range(len(center_pos_list)):
        tmp = RectangleWall(center_pos_list[i_w], hExt_list[i_w])
        recWall_list.append(tmp)

    tmp_grp = RectangleWallGroup(recWall_list)
    is_cols = []
    for i_r in range(len(r_traj)):
        is_col = tmp_grp.is_point_inside_wg(r_traj[i_r], min_to_wall_dist)
        ## no collision after interploate
        if i_r != len(r_traj) - 1:
            for st in interp_states(r_traj[i_r], r_traj[i_r+1], ):
                is_col = tmp_grp.is_point_inside_wg(st, min_to_wall_dist)
                if is_col: 
                    ## detect collision
                    break
        is_cols.append(is_col)
    return np.array(is_cols)

def interp_states(state, new_state):
    '''
    interpolate between two states numpy 1D e.g., (2,)
    '''
    disp = new_state - state
    collision_eps = 1e-2 # 8e-3
    d = np.linalg.norm( disp )
    K = int(d / collision_eps)
    dense_st = []
    for k in range(0, K):
        c = state + k * 1. / K * disp
        dense_st.append(c)
    return dense_st


def get_is_connected(traj):
    cond_1 = np.abs( traj[-2] - traj[-1] ).sum() < 0.12 * traj.shape[1]
    cond_2 = np.abs( traj[0] - traj[1] ).sum() < 0.12 * traj.shape[1]
    return ( cond_1 and cond_2 )

def compute_dist_sum(traj: np.ndarray):
    assert traj.ndim == 2
    diff = traj[1:] - traj[:-1] # (len - 1, dim)
    diff = np.linalg.norm( diff, axis=1 )
    return diff.sum().item()
    

def pad_traj2d(traj: np.ndarray, req_len, pad_type='last'):
    ''' (t, 2) -> (t+res, 2), last pad '''
    residual = req_len - traj.shape[0]
    assert residual >= 0
    if residual > 0:
        pad = traj[-1:, :].repeat(residual, axis=0) # (1, 2) -> (res, 2)
        traj = np.append( traj, pad, axis=0 ) # (t+res, 2)
    return traj

def pad_traj2d_list(env_solutions: List[np.ndarray]):
    '''given a list of traj with different horizon, pad them to the max horizon'''
    # list of np2d
    assert env_solutions[0].ndim == 2
    max_len = max([ len(s) for s in env_solutions ])
    tmp = [] # new list of np
    for i in range(len(env_solutions)):
        tmp.append( pad_traj2d(env_solutions[i], max_len) )
    return tmp

def pad_traj2d_list_v2(env_solutions: List[np.ndarray], req_len):
    '''pad them to the given horizon 
    shape [ (t, 2), ] (given a list of traj with different horizon)'''
    # list of np2d
    assert env_solutions[0].ndim == 2
    tmp = [] # new list of np
    for i in range(len(env_solutions)):
        tmp.append( pad_traj2d(env_solutions[i], req_len) )
    return tmp

def pad_traj2d_list_v3(env_solutions: List[np.ndarray], target:List[np.ndarray]):
    '''pad them to the given horizon 
    shape [ (t, 2), ] (given a list of traj with different horizon)'''
    # list of np2d
    assert env_solutions[0].ndim == 2
    assert target[0].ndim == 2
    tmp = [] # new list of np
    for i in range(len(env_solutions)):
        tmp.append( pad_traj2d(env_solutions[i], len(target[i])) )
    return tmp

def round_down_to_nearest(num, denom: int):
    return denom * np.floor(num / denom)

def round_to_nearest(num, denom: int) -> np.ndarray:
    return denom * np.round(num / denom)

def split_cont_traj(env_traj, env_done, n_vis=20):
    ''' given a long consectutive traj, return a list of episode
    '''
    done_idx = np.where(env_done)[0][:n_vis] # np1d
    trajs = np.split( np.copy(env_traj), done_idx )[:-1] # list of len: n_vis
    return trajs

def plt_img(img, dpi=100):
    plt.clf(); plt.figure(dpi=dpi); plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow(img)
    plt.tight_layout()
    plt.margins(x=0)
    plt.margins(y=0)
    plt.show()

def save_img(savepath: str, img: np.ndarray):
    imageio.imsave(savepath, img)
    



if __name__ == '__main__':
    maze_size = np.array([5,5])
    # xy = np.array([[0.7, 0.1], [2.64, 0.93],[1.93, 4.7 ], [0.93, 1.13], [0.8 , 4.08],[0.82, 4.84],])
    xy = np.array([[2.64, 4.93], [0.7, 0.1]])
    xy = np.array([[0.7, 0.1], [2.64, 4.93],[0.5,5.5] ])
    spl = SplineInterp().fit_bspline(xy, 20, maze_size, True)