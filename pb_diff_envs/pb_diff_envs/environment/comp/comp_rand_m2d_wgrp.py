import numpy as np
import pdb
import os
from pb_diff_envs.environment.static.comp_rec_group import ComposedRandRecGrp 
from pb_diff_envs.environment.static.comp_rm2d_env import ComposedRM2DEnv
from pb_diff_envs.environment.static.maze2d_rand_wgrp import Maze2DRandRecGroupList

class ComposedRM2DGroupList(Maze2DRandRecGroupList):
    def __init__(self, robot_env, num_groups, n_comp,
                 num_walls_c, 
                 maze_size: np.ndarray, hExt_range_c, comp_type, is_static=True, seed=270, **kwargs):
        
        assert robot_env == ComposedRM2DEnv

        self.n_comp = n_comp
        self.hExt_range_c = hExt_range_c
        self.num_walls_c = num_walls_c
        self.comp_type = comp_type
        self.grp_seed_diff = kwargs['mazelist_config']['grp_seed_diff']
        self.r_cfg = dict(num_walls_c=num_walls_c)

        super().__init__(robot_env, num_groups, num_walls_c, maze_size,
                          hExt_range_c[0], is_static, seed, **kwargs)
        del self.num_walls
        del self.hExt_range


    def setup_wgrp(self):
        tmp_ng = self.num_groups if not self.is_eval else self.kwargs['eval_num_groups']
        self.rand_rec_group = ComposedRandRecGrp # could be a composed group

        self.recs_grp = self.rand_rec_group(
            env_list_name=self.env_name,
            num_groups=tmp_ng,
            n_comp=self.n_comp,
            num_walls_c=self.num_walls_c,
            maze_size=self.maze_size,
            half_extents_c=self.hExt_range_c,
            gap_betw_wall=self.gap_betw_wall,
            seed=self.rng_seed,
            gen_data=self.gen_data,
            ##
            robot_class=self.robot_env.robot_class,
            is_eval=self.is_eval,
            rand_iter_limit=self.mazelist_config.get('rand_iter_limit', 1e5),
            grp_seed_diff=self.grp_seed_diff,
            mazelist_config=self.mazelist_config,
        )


        self.wallLoc_list = self.recs_grp.wallLoc_list.copy() # [ng, nw1+nw2, 2]
        self.hExt_list = self.recs_grp.hExt_list.copy() # [ng, nw1+nw2, 2]


        



    def create_single_env(self, env_idx) -> ComposedRM2DEnv:
        '''put the created env to model_list, also return the same env instance'''
        if self.model_list[env_idx] is None:

            wall_locations_c, wall_hExts_c = self.recs_grp.get_composed_item(env_idx)

            env = self.robot_env(wall_locations_c, wall_hExts_c,
                                 self.robot_config,
                                     renderer_config=self.r_cfg,  **self.mazelist_config)
            self.model_list[env_idx] = env

        # unload prev env
        # if env_idx > 0 and self.model_list[env_idx-1] is not None:
            # self.model_list[env_idx-1].unload_env()
        return self.model_list[env_idx]
    
    def create_env_by_pos(self, env_idx, wall_locations_c, wall_hExts_c):
        '''set the env_idx to the specific user given env'''
        # wall_hExts_c.ndim == 3 and wall_hExts_c.ndim == 3 and 
        # assert isinstance(wall_locations_c, np.ndarray)

        env = self.robot_env(wall_locations_c, wall_hExts_c,
                                 self.robot_config,
                                     renderer_config=self.r_cfg,  **self.mazelist_config)
        self.model_list[env_idx] = env

        return self.model_list[env_idx]