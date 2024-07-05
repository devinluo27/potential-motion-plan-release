import numpy as np
import pdb, os
from pb_diff_envs.environment.rand_rec_group_43 import RandRectangleGroup_43
from pb_diff_envs.environment.static.maze2d_rand_wgrp import Maze2DRandRecGroupList

class Maze2DRandRecGroupList_43(Maze2DRandRecGroupList):
    '''
    Group of envs with concave obstacles, each concave obstacle consists of 3 small square, for example,
    X    XX   XX
    XX , X  ,  X,
    each layout corresponds to a different mode, so the representation of each obstacle: [x, y, mode_idx]
    '''
    def __init__(self, robot_env, num_groups, num_walls, maze_size: np.ndarray, hExt_range, is_static=True, seed=270, **kwargs):
        super().__init__(robot_env, num_groups, num_walls, maze_size, hExt_range, is_static, seed, **kwargs)
        self.env_type = 'static_maze2d_concave_c43'
        self.recs_grp.save_center_and_mode()

    def setup_wgrp(self):
        tmp_ng = self.num_groups if not self.is_eval else self.kwargs['eval_num_groups']
        self.rand_rec_group = RandRectangleGroup_43 # could be a composed group

        self.recs_grp = self.rand_rec_group(
            env_list_name=self.env_name,
            num_groups=tmp_ng,
            num_walls=self.num_walls,
            maze_size=self.maze_size,
            half_extents=self.hExt_range,
            gap_betw_wall=self.gap_betw_wall,
            seed=self.rng_seed,
            gen_data=self.gen_data,

            robot_class=self.robot_env.robot_class,
            is_eval=self.is_eval,
            rand_iter_limit=self.mazelist_config.get('rand_iter_limit', 1e5)
        )

        # (ng, n_v, 2), e.g. (5000, 6, 2), not to big
        # hExt could be different in composing cases
        self.wallLoc_list = np.copy(self.recs_grp.rec_loc_grp_list)
        self.hExt_list = np.copy(self.recs_grp.rec_hExt_grp_list)




