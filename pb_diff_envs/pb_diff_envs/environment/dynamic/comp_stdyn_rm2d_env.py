import numpy as np
from pb_diff_envs.robot.point2d_robot import Point2DRobot
from pb_diff_envs.environment.offline_env import OfflineEnv
from pb_diff_envs.environment.pybullet_env import PybulletEnv
from pb_diff_envs.environment.abs_maze2d_env import AbsDynamicMaze2DEnv
from pb_diff_envs.environment.dynamic_rrgroup import DynamicRectangleWall, DynamicRecWallGroup
from pb_diff_envs.objects.trajectory import WaypointLinearTrajectory
from typing import List
from gym.utils import EzPickle
from pb_diff_envs.utils.maze2d_utils import get_is_collision_wtrajs
from .dyn_wtraj_gen import DynWallTrajGenerator
from .dyn_rm2d_renderer import DynamicRecRenderer
from pb_diff_envs.planner.sipp_planner import SippPlanner
from .dyn_rm2d_env import DynamicRandMaze2DEnv

class ComposedStDynRandMaze2DEnv(DynamicRandMaze2DEnv, OfflineEnv, PybulletEnv, EzPickle): 

    robot_class = Point2DRobot

    def __init__(self, wall_locations, wall_hExts, wall_is_dyn: List[bool], 
                 len_full_wtraj: int, # not used
                 num_sample_pts: int,
                 gap_betw_wall: int,
                robot_config, wp_seed, speed, renderer_config={}, **kwargs):
        '''
        wall_locations: only the start location, keep changing
        current wall_locations is in the DynamicRecWallGroup
        num_sample_pts: num of wtraj rand points before interp, include start end
        '''
        

        assert wall_is_dyn[0] and not wall_is_dyn[-1], 'dyn wall first, then static walls'
        self.wall_is_dyn = wall_is_dyn
        
        super().__init__(
            wall_locations, wall_hExts,
                 len_full_wtraj,
                 num_sample_pts,
                 gap_betw_wall,
                robot_config, wp_seed, speed, renderer_config, **kwargs
        )
        self.env_id = 'comp_static_dynamic_maze2d'





    def update_dyn_wallgroup(self):
        '''init wall group for the collision checking
        wtraj_start_wlocs, wtraj_dest_wlocs:
        we keep the start and dest location of the current dr_group
        '''
        # assert not hasattr(self, 'recWall_grp') ,'only called once'
        if hasattr(self, 'recWall_grp'):
            del self.recWall_grp
        # 1. get new wtraj
        wtrajs = self.get_rand_wtrajs()


        self.cur_wtrajs = [] # wtrajs # to be checked
        print(f'[dyn env wtrajs] {wtrajs.shape}')
        recWall_list = []
        ## 2. inject wtraj to grp
        for i_w in range(self.num_walls):
            if self.wall_is_dyn[i_w]:
                wj = wtrajs[i_w]
                wpl_traj = WaypointLinearTrajectory(wj)
            else:
                wj = np.array( self.wtraj_start_wlocs[i_w] )[None,].repeat( len(wtrajs[0]), axis=0)
                wpl_traj = WaypointLinearTrajectory(wj)
            
            self.cur_wtrajs.append(wj)

            tmp = DynamicRectangleWall(self.wtraj_start_wlocs[i_w], 
                                       self.wall_hExts[i_w], wpl_traj)
            recWall_list.append(tmp)

        self.cur_wtrajs = np.stack( self.cur_wtrajs, axis=0 )


        self.recWall_grp = DynamicRecWallGroup(recWall_list)
        ## For SIPP, abstraction
        self.objects = self.recWall_grp.dyn_recWall_list

        self.robot.recWall_grp = self.recWall_grp
        # self.robot = Point2DRobot(recWall_grp=self.recWall_grp, **self.robot_config)

        # 3. 
        # we assume that we must move the wall to the dest_wlocs before sample another

        self.buffer_wtraj[ self.total_cnt_wtraj % 10] = wtrajs
        self.total_cnt_wtraj += 1



