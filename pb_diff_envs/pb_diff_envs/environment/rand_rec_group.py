import matplotlib.pyplot as plt
import random
from typing import List
import numpy as np
from tqdm import tqdm
import os
from pb_diff_envs.utils.kuka_utils_luo import from_rel2abs_path
import pdb

class RandRectangleGroup:
    '''
    Construct a group of obstacles in Rectangle Shape, each obstacle is represented in x-y location and half extent
    '''
    def __init__(self, env_list_name:str, num_groups, num_walls, 
                 maze_size, half_extents, gap_betw_wall, seed, gen_data,
                 robot_class, is_eval=False,
                 **kwargs) -> None:
        
        self.rng = np.random.default_rng(seed)
        self.rng_color = np.random.default_rng(100)

        self.env_list_name = env_list_name
        self.num_groups = num_groups
        self.num_walls = num_walls # num_walls

        self.maze_size = maze_size # np(5, 5)
        self.half_extents = np.copy(half_extents) # tuple of size 3? 2 -> np
        self.gap_betw_wall = gap_betw_wall # prevent a separate divider
        assert half_extents.shape == (2,) and half_extents[0] == half_extents[1]

        self.wall_size = half_extents * 2 # tuple of size 3
        self.base_orn = [0, 0, 0, 1]
        self.sort_wloc = True # sort the order as other env does


        self.gen_data = gen_data # if False, directly load
        self.GUI = kwargs.get('GUI', False)
        self.debug_mode = kwargs.get('debug_mode', False)
        # self.robot_class = robot_class
        # self.load_vis_objects = True # if load robot and table
        self.is_eval = is_eval
        self.rand_iter_limit = kwargs['rand_iter_limit'] # default 1e5

        if self.gen_data:
            self.create_envs()
        else:
            self.rec_loc_grp_list = self.load_rec_loc_grp_list()
            rec_hExt_grp = np.stack([self.half_extents,] * self.num_walls, axis=0)
            self.rec_hExt_grp_list = rec_hExt_grp[None,].repeat(self.num_groups ,axis=0)
            # pdb.set_trace()
            assert self.rec_hExt_grp_list.shape == (self.num_groups, self.num_walls, 2)

    def create_envs(self):
        '''set rec_loc_grp_list'''

        rec_loc_grp_list: list[np.ndarray] = [] # a list of np2d: n_c,3
        rec_hExt_grp_list = []
        for _ in tqdm(range(self.num_groups)):
            recloc_grp, rec_hExt_grp = self.sample_one_valid_env()
            rec_loc_grp_list.append(recloc_grp) # center location, np2d
            rec_hExt_grp_list.append(rec_hExt_grp) # np2d
        
        # np3d: ng, n_cube, 2
        self.rec_loc_grp_list = np.array(rec_loc_grp_list)
        self.rec_hExt_grp_list = np.array(rec_hExt_grp_list)

        self.save_rlg_list()

        return
    


    def sample_one_valid_env(self):
        '''create one env config without any overlap
        returns:
        array (n_c, 3) of 3d position '''
        # self.reset()
        rec_list = []
        cnt_iter = 0
        while len(rec_list) < self.num_walls:
            cnt_iter += 1
            if cnt_iter > self.rand_iter_limit: # deadlock, reset everything
                # self.reset()
                print('reach limit reset')
                for rec in rec_list: 
                    print(rec.center)
                rec_list = []
                cnt_iter = 0

            center_pos, hExt = self.sample_xy_hExt()
            tmp_rec = RectangleWall(center_pos, hExt)
            has_overlap = self.check_is_overlap(rec_list, tmp_rec)

            if has_overlap:
                pass
            else:
                # pos_list.append(center_pos.tolist()) # for sorting
                # hExt_list.append(hExt)
                rec_list.append(tmp_rec)


        # print('cnt_iter', cnt_iter) # usually < 1000
        # self.reset()
        # pre sorted
        pos_list, hExt_list = self.get_pos_hExt_list(rec_list)
        # print('1 pos_list', pos_list)
        if self.sort_wloc:
            ## pos_list must be 2d, a list of list
            idx_and_pos = sorted(enumerate(pos_list), key=lambda i:i[1]) # list of a tuple (idx, w)
            pos_list = [i_p[1] for i_p in idx_and_pos] # i_p a tuple, still a list of list
            pos_sort_idx = [i_p[0] for i_p in idx_and_pos] # a list of int
            
            hExt_list = np.array(hExt_list)[pos_sort_idx] # switch order correspondingly
        else:
            assert False

        pos_list = np.array(pos_list) # n_c, 2
        # print('2 pos_list', pos_list)
        return pos_list, hExt_list
    

    # ------------- helpers for sampling ----------------
    def sample_xy_hExt(self):
        '''simply sample a location in all valid range, not consider overlap yet'''
        hExt = self.half_extents.copy()
        bottom_left, top_right = self.get_rec_center_range(hExt)
        while True:
            center_pos = self.rng.uniform(low=bottom_left, high=top_right) # (2,)
            # print('center_pos', center_pos)
            if False: # self.is_in_void_range(center_pos, half_extents=hExt):
                continue
            else:
                break
        return center_pos, hExt


    def get_rec_center_range(self, hExt):
        bottom_left = 0 + hExt + self.gap_betw_wall
        top_right = self.maze_size - hExt - self.gap_betw_wall
        return bottom_left, top_right
    
    def check_is_overlap(self, rec_list, rec_new):
        for rec in rec_list:
            is_ovlp = is_recWall_overlap(rec, rec_new, self.gap_betw_wall)
            if is_ovlp: # overlap
                return True
        return False

    def get_pos_hExt_list(self, rec_list): # List[RectangleWall]
        '''from a list of rectangle to a list of list'''
        pos_list = []
        hExt_list = []
        for rec in rec_list:
            pos_list.append(rec.center.tolist())
            hExt_list.append(rec.hExt.tolist())
        return pos_list, hExt_list





    # --------- save and load the wall locations -----------
    def get_npyname(self):
        self.prefix = from_rel2abs_path(__file__, '../datasets/rand_rec2dgrp/')
        os.makedirs(self.prefix, exist_ok=True)
        if self.is_eval:
            npyname = f'{self.prefix}/{self.env_list_name}_eval.npy'
        else:
            npyname = f'{self.prefix}/{self.env_list_name}.npy'
        return npyname
    
    ## np (num_groups,3)
    def save_rlg_list(self):
        assert self.rec_loc_grp_list.shape[0] == self.num_groups
        npyname = self.get_npyname()
        ## check instead
        if os.path.exists(npyname) and 'testOnly' not in npyname:
            # assert False
            self.check_matched()
        else:
            np.save(npyname, self.rec_loc_grp_list)
            if 'testOnly' not in npyname:
                os.chmod(npyname, 0o444)

    def load_rec_loc_grp_list(self):
        rec_loc_grp_list = np.load(self.get_npyname())
        return rec_loc_grp_list


    def check_matched(self):
        
        assert (self.rec_loc_grp_list == self.load_rec_loc_grp_list()).all()

    # ----------- maybe some code for checking and visualization ---------------


class RectangleWall:
    '''a single rectangle, like one wall'''
    def __init__(self, center: np.ndarray, hExt: np.ndarray) -> None:
        self.center = np.copy(center)
        self.hExt = np.copy(hExt)
        self.top_right = center + hExt
        self.bottom_left = center - hExt

    def update_center_pos(self, center):
        '''used for dynamic wall'''
        self.center = center
        self.top_right = center + self.hExt
        self.bottom_left = center - self.hExt
        # pdb.set_trace()

    def set_config(self, center):
        assert type(center) == np.ndarray and center.shape == (2,)
        self.update_center_pos(center)

    def is_point_inside(self, pose:np.ndarray, min_to_wall_dist:float):
        '''True if collision, input is unnormed'''
        # pose (2,)
        cond_1 = ( pose > ( self.bottom_left -  min_to_wall_dist) ).all()
        cond_2 = ( pose < ( self.top_right + min_to_wall_dist) ).all()
        return cond_1 & cond_2


class RectangleWallGroup:
    '''
    a group of rectangles, typically as *one* env
    '''
    def __init__(self, recWall_list: List[RectangleWall]) -> None:
        self.recWall_list = recWall_list
    
    def is_point_inside_wg(self, pose, min_to_wall_dist):
        '''min_to_wall_dist: if 0.01, cannot be in region nearer than 0.01'''
        for recWall in self.recWall_list:
            is_col = recWall.is_point_inside(pose, min_to_wall_dist)
            if is_col:
                return True
        return False




def is_recWall_overlap(rec_1, rec_new: RectangleWall, gap):
    ''' True if there is overlap
    make sure that rec_1 is the one in the maze,
    rec_2 is the new one to check,
    we enlarge rec_1 a little bit
    '''
    return not (rec_1.top_right[0] + gap  < rec_new.bottom_left[0]
            or rec_1.bottom_left[0] - gap > rec_new.top_right[0]
            or rec_1.top_right[1] + gap   < rec_new.bottom_left[1]
            or rec_1.bottom_left[1] - gap > rec_new.top_right[1])

