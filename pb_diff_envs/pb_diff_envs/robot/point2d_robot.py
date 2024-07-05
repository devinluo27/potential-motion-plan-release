from typing import Union
import numpy as np
from pb_diff_envs.robot.abstract_robot import AbstractRobot
from pb_diff_envs.environment.rand_rec_group import RectangleWallGroup
from pb_diff_envs.environment.dynamic_rrgroup import DynamicRecWallGroup
import pybullet as p
import os.path as osp
import pdb

class AbsPointRobot(AbstractRobot):
    '''place holder robot class, 
    ensure no pybullet involve'''
    
    def load(self, **kwargs):
        assert False, 'no need'
        
    def load2pybullet(self, **kwargs):
        assert False, 'no need'

class Point2DRobot(AbsPointRobot):
    def __init__(self, maze_size, recWall_grp, min_to_wall_dist,
                collision_eps, **kwargs):
        """
        limits_low, limits_high: (np1d [2,]) according to maze size
        min_to_wall_dist: used when compute collision
        """
        limits_low = np.array([0.,0.], dtype=np.float32)
        limits_high = np.copy(maze_size).astype(np.float32)
        super().__init__(limits_low, limits_high, collision_eps, **kwargs)

        self.cur_pose = np.zeros(shape=(2,), dtype=np.float32) + 1e-3 # init_pose
        # ---------- setup -----------
        self.maze_size = maze_size
        self.recWall_grp: Union[RectangleWallGroup, DynamicRecWallGroup] = recWall_grp # or DynamicRecWallGroup
        self.min_to_wall_dist = min_to_wall_dist # 0.01
        self.collision_check_count = 0 # Not used, placeholder
        self.joints_max_force = np.ones_like(limits_low)

        assert limits_low.shape == (2,) and maze_size.shape == (2,)
        assert min_to_wall_dist <= 0.02
    
    def set_config(self, config, item_id=None):
        assert (config <= self.limits_high).all()
        assert (config >= self.limits_low).all()

        self.cur_pose = config # np
        return
    

    def no_collision(self):
        has_col = self.recWall_grp.is_point_inside_wg(self.cur_pose, self.min_to_wall_dist)
        self.collision_check_count += 1

        return not has_col
    
    def no_collision_0dist(self):
        '''used in evaluation'''
        has_col = self.recWall_grp.is_point_inside_wg(self.cur_pose, 0)
        self.collision_check_count += 1
        return not has_col
    
