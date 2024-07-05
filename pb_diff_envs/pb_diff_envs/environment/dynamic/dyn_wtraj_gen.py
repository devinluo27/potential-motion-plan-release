import numpy as np
from pb_diff_envs.utils.maze2d_utils import SplineInterp, pad_traj2d, round_down_to_nearest
import matplotlib.pyplot as plt

class DynWallTrajGenerator:
    def __init__(self, maze_size, len_full_wtraj, num_sample_pts, gap_betw_wall, 
                    w_speed, min_wtraj_distance=None):
        '''
        generate a wall np trajectory for Maze 2D Env
        we must assume that the step size of wtraj are same
        '''
        self.maze_size = maze_size
        self.len_full_wtraj = len_full_wtraj # pts after interp
        assert num_sample_pts >= 2
        self.num_sample_pts = num_sample_pts # pts before interp
        self.gap_betw_wall = gap_betw_wall # just not too close to the maze 2boundary
        # self.wtraj_len = wtraj_len
        self.w_speed = w_speed
        if num_sample_pts == 2: # one line
            pass
        else:
            raise NotImplementedError()


        self.round_down = True
        self.spl_interp = SplineInterp()
        self.eps_min_dist = 0.3
        self.sample_cnt_limit = 1e4

    def get_rand_linear_wtrajs(self, prev_wpose_list, hExt_list, rng, pad_type='last'):
        assert pad_type in ['last', None]
        wtrajs = []
        ## 1. get traj of different len
        for i_w in range(len(prev_wpose_list)):
            prev_wpose = prev_wpose_list[i_w]
            wtraj = self.get_rand_linear_dynWall_traj(prev_wpose, hExt_list[i_w], rng)
            wtrajs.append(wtraj)
        
        ## 2. pad all trajs to the max len
        if pad_type == 'last':
            max_len = max ( [ wtrajs[i].shape[0] for i in range(len(wtrajs)) ] )
            for i in range( len(wtrajs) ):
                if wtrajs[i].shape[0] < max_len:
                    wtrajs[i] = pad_traj2d(wtrajs[i], max_len, pad_type)
        
        return np.array(wtrajs)
    
    def get_rand_linear_dynWall_traj(self, prev_wpose, hExt, rng, wlen=None):
        '''return an interpolated random traj of (n,2)
        wlen: len in real world unit, e.g. 0-5*1.4
        '''
        assert prev_wpose.shape == (2,) and hExt.shape == (2,)
        assert self.num_sample_pts == 2
        self.min_wtraj_distance = self.get_wtraj_len_min(hExt)

        rand_pts = self.get_rand_pts(prev_wpose, hExt, rng)[0] # take the first pt

        # print(f'min wtraj: {self.min_wtraj_distance}') # ms55: 2.4 if hExt1; 3.5 if hExt0.5
        # if wlen is not None:
            # assert wlen > self.min_wtraj_distance
            # rand_pts = prev_wpose + wlen * (rand_pts - prev_wpose) / np.linalg.norm(rand_pts - prev_wpose)
        
        if self.round_down:
            diff = np.linalg.norm(rand_pts - prev_wpose)
            unit_vector = (rand_pts - prev_wpose) / diff
            diff = round_down_to_nearest(diff, self.w_speed)
            rand_pts = prev_wpose + diff * unit_vector
        
        # (n, 2)
        # way_pts = np.concatenate([prev_wpose[None,], rand_pts], axis=0)
        # maybe we need a fixed point traj
        # rand_traj = self.spl_interp.fit_bspline(way_pts, self.len_full_wtraj, self.maze_size, True)
        ## has made sure devisible upward
        rand_traj = self.spl_interp.fit_line_same_stepsize(prev_wpose, rand_pts, self.w_speed, is_plot=False)

        return rand_traj 

    def get_rand_pts(self, prev_wpose, hExt, rng: np.random.Generator):
        ''' return random way points
        enforce min distance to every sample point '''
        bottom_left, top_right = self.get_rec_center_range(hExt)
        pt_list = [] # list of np1d 2,
        for _ in range(self.num_sample_pts - 1):
            cnt_iter = 0
            while True:
                cnt_iter += 1
                rand_pt = rng.uniform(low=bottom_left, high=top_right, size=(2,) )
                diff = np.linalg.norm(prev_wpose - rand_pt)
                r_limit = cnt_iter > self.sample_cnt_limit
                tmp = (diff > self.min_wtraj_distance) or r_limit
                if tmp:
                    if r_limit: # not elegant
                        rand_pt = bottom_left if rng.uniform() > 0.5 else top_right
                    # print(f'wtraj gen diff: {diff}; r_limit: {r_limit}', flush=True)
                    pt_list.append(rand_pt)
                    prev_wpose = np.copy(rand_pt)
                    break # inner while
        
        return np.array(pt_list)




    def get_wtraj_len_limit(self, hExt):
        # assume square: (ms/2 - gap - hExt) * sqrt(2)
        ## skew line
        return 2 * (self.maze_size[0] / 2 - self.gap_betw_wall - hExt[0]) * 1.4
    
    def get_wtraj_len_min(self, hExt):
        # assume square: a default min len to prevent trivial cases
        ## edge len
        return self.maze_size[0] - (self.gap_betw_wall + hExt[0]) * 2 - self.eps_min_dist

    def get_rec_center_range(self, hExt):
        '''same as in rand rec group'''
        bottom_left = 0 + hExt + self.gap_betw_wall
        top_right = self.maze_size - hExt - self.gap_betw_wall
        return bottom_left, top_right
    



    def get_rand_dynWall_traj_luotest(self):
        ''' naive demo
        return an interpolated random traj of (n,2)'''
        # range [0, 0] - [5, 5]
        rand_pts = np.random.rand(self.num_sample_pts, 2) * self.maze_size
        rand_traj = self.spl_interp.fit_bspline(rand_pts, self.len_full_wtraj, self.maze_size, True) 
        return rand_traj 