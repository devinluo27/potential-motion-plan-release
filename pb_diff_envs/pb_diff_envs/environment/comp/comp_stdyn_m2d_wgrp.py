import numpy as np
import pdb
import os, copy
from pb_diff_envs.environment.dynamic.comp_stdyn_rm2d_env import ComposedStDynRandMaze2DEnv
from pb_diff_envs.environment.comp.comp_rand_m2d_wgrp import ComposedRM2DGroupList
from pb_diff_envs.environment.dynamic.dyn_rm2d_wgrp import DynRM2DGroupList

class ComposedStDynRM2DGroupList(ComposedRM2DGroupList, DynRM2DGroupList):
    def __init__(self, robot_env, num_groups, n_comp,
                 num_walls_c, 
                 maze_size: np.ndarray, hExt_range_c, num_sample_pts, speed, comp_type, is_static=False, seed=270, **kwargs):
        
        assert robot_env == ComposedStDynRandMaze2DEnv

        self.n_comp = n_comp
        self.hExt_range_c = hExt_range_c
        self.num_walls_c = num_walls_c
        self.comp_type = comp_type
        self.grp_seed_diff = kwargs['mazelist_config']['grp_seed_diff']
        self.wall_is_dyn = kwargs['wall_is_dyn']
        self.r_cfg = dict(num_walls_c=num_walls_c)

        
        DynRM2DGroupList.__init__(self,           
                robot_env, num_groups, num_walls_c,
                maze_size, 
                hExt_range_c[0], num_sample_pts, speed,
                is_static, seed,
                **kwargs)

        self.num_walls = num_walls_c.sum().item()
        del self.hExt_range


    def create_single_env(self, env_idx) -> ComposedStDynRandMaze2DEnv:
        '''put the created env to model_list, also return the same env instance'''
        if self.model_list[env_idx] is None:

            # wall_locations_c, wall_hExts_c = self.recs_grp.get_composed_item(env_idx)
            # wall_locations_c is a list of np (nw, 2), e.g. [(6, 2), (3, 2)]
            
            tmp1 = self.recs_grp.wallLoc_list[env_idx] # (nw1+nw2, 2)
            tmp2 = self.recs_grp.hExt_list[env_idx]

            ## automatic pass in planner_timeout, seed_planner
            
            tmp_config = copy.deepcopy(self.mazelist_config)
            del tmp_config['gap_betw_wall']
            env = self.robot_env(tmp1, tmp2,
                 self.wall_is_dyn, # list of bool
                 None, 
                 self.num_sample_pts,
                 self.gap_betw_wall,
                 self.robot_config, 
                 self.wp_seed, 
                 self.speed, 
                 renderer_config=self.r_cfg, **tmp_config)

            self.model_list[env_idx] = env

        # unload prev env
        # if env_idx > 0 and self.model_list[env_idx-1] is not None:
            # self.model_list[env_idx-1].unload_env()
        return self.model_list[env_idx]