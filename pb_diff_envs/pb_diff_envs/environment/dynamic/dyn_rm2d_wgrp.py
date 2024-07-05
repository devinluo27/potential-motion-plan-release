import numpy as np
from pb_diff_envs.environment.dynamic.dyn_rm2d_env import DynamicRandMaze2DEnv
from pb_diff_envs.environment.dynamic.comp_stdyn_rm2d_env import ComposedStDynRandMaze2DEnv
import copy, os
from ..static.maze2d_rand_wgrp import Maze2DRandRecGroupList

class DynRM2DGroupList(Maze2DRandRecGroupList):
    def __init__(self, robot_env, num_groups, num_walls, 
                maze_size: np.ndarray, 
                hExt_range, num_sample_pts, speed,
                is_static=False, seed=270, 
                **kwargs):
        print(robot_env)
        assert robot_env in [DynamicRandMaze2DEnv, ComposedStDynRandMaze2DEnv]
        assert num_sample_pts == 2
        self.num_sample_pts = num_sample_pts # for w_traj
        self.speed = speed

        self.wp_seed = kwargs.get('wp_seed', 100)
        print(f'dyn wgrp wpseed: {self.wp_seed}')

        super().__init__(robot_env, num_groups, num_walls, 
                         maze_size, hExt_range, is_static, seed, **kwargs)
        self.env_type = 'dyn_maze2d'
        
        
    def create_single_env(self, env_idx) -> DynamicRandMaze2DEnv:
        '''put the created env to model_list, also return the same env instance'''
        if self.model_list[env_idx] is None:
            wall_locations = self.wallLoc_list[env_idx]
            wall_hExts = self.hExt_list[env_idx]
            tmp_config = copy.deepcopy(self.mazelist_config)
            del tmp_config['gap_betw_wall']
            env = self.robot_env(wall_locations, wall_hExts, 
                 None, 
                 self.num_sample_pts,
                 self.gap_betw_wall,
                 self.robot_config, 
                 self.wp_seed, 
                 self.speed, 
                 renderer_config={}, **tmp_config)

            self.model_list[env_idx] = env

        ## here no need to unload prev env (but pybullet env should do)
        return self.model_list[env_idx]
    
    def render_mazelist(self, savepath, r_traj_list, wtrajs_list):
        # input list of numpy
        env0_single = self.create_single_env(0)
        # render every traj
        for i in range(len(r_traj_list)):
            assert r_traj_list[i].ndim == 2
            assert wtrajs_list[i].ndim == 3, 'h, nw, 2'

            fn_base, fn_ext = os.path.splitext(savepath) # xx/xx/xx/, xx.gif
            fn = f"{fn_base}_{i}.gif"
            env0_single.render_1_traj(r_traj_list[i], wtrajs_list[i], img_type='gif',
                                      savepath=fn)
            



