import matplotlib.pyplot as plt
import random
from typing import List
import numpy as np
from tqdm import tqdm
import os
from pb_diff_envs.utils.kuka_utils_luo import from_rel2abs_path
from pb_diff_envs.environment.rand_rec_group import RectangleWall
from pb_diff_envs.objects.trajectory import WaypointLinearTrajectory
import pdb, time

class AbsDynamicWall:
    pass

class DynamicRectangleWall(AbsDynamicWall):
    '''
    a single rectangle, e.g., one wall/obstacle
    '''
    def __init__(self, center: np.ndarray, hExt: np.ndarray, 
                 trajectory: WaypointLinearTrajectory) -> None:
        # RectangleWall.__init__(self, center, hExt)
        self.item = RectangleWall(center, hExt)
        self.trajectory = trajectory
        assert (center - trajectory.waypoints[0] < 1e-3).all()
        

    def set_config_at_time(self, t):
        ## maybe linear interp, depends on traj class type 
        spec = self.trajectory.get_spec(t)
        # set the spec of time t to item, this will call the set_config below
        self.trajectory.set_spec(self.item, spec)




class DynamicRecWallGroup:
    '''
    a group of rectangles, typically as *one* env
    it has a list of 
    '''
    def __init__(self, dyn_recWall_list: List[DynamicRectangleWall]) -> None:
        self.dyn_recWall_list = dyn_recWall_list
        # self.len_wtraj = dyn_recWall_list[0].trajectory.waypoints.shape[0] # all wtraj must be the same
        self.check_wtrajs_same_len()
    
    def get_max_len_wtraj(self):
        '''get the max len of trajs in the group'''
        max_len = max([ len(obj.trajectory.waypoints) for obj in self.dyn_recWall_list ])
        return max_len
    
    def get_wtrajs(self):
        '''return the traj of all walls in the group
        assume all wtraj has same length
        returns: np3d (n_w, n_pts, 2)'''
        wtrajs = []
        for dyn_wall in self.dyn_recWall_list:
            wtrajs.append(  dyn_wall.trajectory.waypoints  )
            if len(wtrajs) > 0:
                assert wtrajs[0].shape == wtrajs[-1].shape, 'ensure all wtraj same len'
        
        return np.array(wtrajs)


    def is_point_inside_wg(self, pose, min_to_wall_dist):
        ''' NOTE used in robot collision checking
        ** the inner wall instance must be updated to latest'''

        for dr_wall in self.dyn_recWall_list:
            # print('wloc', dr_wall.item.center, dr_wall.item.bottom_left)

            is_col = dr_wall.item.is_point_inside(pose, min_to_wall_dist)
            if is_col:
                return True
        return False


    def set_walls_at_time(self, timestep: float):
        '''NOTE this func will change wall Loc'''
        # assert type(timestep) == int
        for obj in self.dyn_recWall_list:
            obj.set_config_at_time(timestep)

    
    
    # ------------ utils ---------------

    def is_walls_in_start_pos(self):
        centers = self.get_walls_pos()
        for i_w in range(len(centers)):
            tmp = ( centers[i_w] - self.dyn_recWall_list[i_w].trajectory.waypoints[0] < 1e-3 ).all()
            if not tmp:
                return False
        return True
    
    def get_wtrajs_end_pos(self):
        ends = []
        for dyn_wall in self.dyn_recWall_list:
            ends.append(dyn_wall.trajectory.waypoints[-1])

        return np.array(ends)

    def get_walls_pos_at_t(self, t):
        '''only return the center poses of give t'''
        centers = []
        for dyn_wall in self.dyn_recWall_list:
            centers.append(dyn_wall.trajectory.get_spec(t))
        return np.array(centers)

    

    def get_walls_pos(self):
        '''only return *current* the center poses'''
        centers = []
        for dyn_wall in self.dyn_recWall_list:
            centers.append(dyn_wall.item.center)
        return np.array(centers)
    
    def get_walls_info(self):
        ''' *current* '''
        centers = []
        hExts = []
        for dyn_wall in self.dyn_recWall_list:
            centers.append(dyn_wall.item.center)
            hExts.append(dyn_wall.item.hExt)
        return np.array(centers), np.array(hExts)
        

    ## ----------- checking ------------
    
    def check_wtrajs_same_len(self):
        len_0 = len(self.dyn_recWall_list[0].trajectory.waypoints)
        for dyn_wall in self.dyn_recWall_list:
            assert len(dyn_wall.trajectory.waypoints) == len_0

    

