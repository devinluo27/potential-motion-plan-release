import numpy as np
import pdb
from typing import Union
from pb_diff_envs.environment.offline_env import OfflineEnv
from pb_diff_envs.environment.pybullet_env import PybulletEnv
import os
from pb_diff_envs.environment.rand_rec_group import RandRectangleGroup
from pb_diff_envs.environment.static.rand_maze2d_env import RandMaze2DEnv
from pb_diff_envs.environment.dynamic.dyn_rm2d_env import DynamicRandMaze2DEnv
from pb_diff_envs.environment.static.comp_rm2d_env import ComposedRM2DEnv
from pb_diff_envs.environment.dynamic.comp_stdyn_rm2d_env import ComposedStDynRandMaze2DEnv
from pb_diff_envs.utils.utils import print_color

class Maze2DRandRecGroupList(OfflineEnv, PybulletEnv):

    def __init__(self, robot_env, num_groups, num_walls, 
                 maze_size: np.ndarray, 
                    
                 hExt_range, is_static=True, seed=270, 
                 **kwargs):
        '''
        NOTE: hExt_range(1d) is not a range, just directly the value for (h,w)
        '''
        self.robot_env = robot_env # a class to call
        assert self.robot_env in [RandMaze2DEnv, DynamicRandMaze2DEnv, 
                                  ComposedRM2DEnv, ComposedStDynRandMaze2DEnv,]


        '''
        when is_eval=True, a different seed is used to generate wall locs,
        for generating val problems.
        '''
        ## 1. -------- setup hyper-param value -----------
        self.is_eval = kwargs.get('is_eval', False)
        self.eval_seed_offset = kwargs.get('eval_seed_offset', 0)
        if self.is_eval:
            assert self.eval_seed_offset != 0
            seed = seed + self.eval_seed_offset

        self.rng_color = np.random.default_rng(100)
        self.rng_seed = seed
        self.num_groups = num_groups
        self.num_walls = num_walls
        self.mazelist_config = kwargs['mazelist_config']
        self.samples_per_env = kwargs['samples_per_env']
        self.min_to_wall_dist = self.mazelist_config['min_to_wall_dist']
        self.gap_betw_wall = self.mazelist_config['gap_betw_wall']



        self.maze_size = np.copy(maze_size) # convert to numpy if is list
        self.hExt_range = np.copy(hExt_range)
        assert hExt_range.shape == (2,)
        self.maze_arr = np.full(shape=maze_size, fill_value=float('-inf')) # placeholder

        # ----- setup config to init robot env ------
        if self.robot_env in [RandMaze2DEnv, ComposedRM2DEnv,]:
            c_eps = self.mazelist_config.get('robot_collision_eps', 0.02)

        elif self.robot_env in [DynamicRandMaze2DEnv, ComposedStDynRandMaze2DEnv]:
            c_eps = self.mazelist_config['robot_collision_eps']
        
        print_color(f'robot_collision_eps: {c_eps}')
        
        self.robot_config = dict(maze_size=self.maze_size,
                                 min_to_wall_dist=self.min_to_wall_dist,
                                 collision_eps=c_eps,
                                 )


        ## 2. ------ setup wall locations ----------
        self.dataset_url = kwargs['dataset_url']
        self.env_name = self.get_env_name()
        # only used when generate the config the first time; otherwise, simply load
        self.gen_data = kwargs.get('gen_data', False)
        
        ## abstraction: customed code
        self.kwargs = kwargs
        self.setup_wgrp() ## wall order in each env should be sorted


        # -------- setup some more values ----------
        # split to how many parts when generating dataset
        self.gen_num_parts = kwargs['gen_num_parts']
        # simply generate env config with another seed
        self.eval_num_groups = kwargs['eval_num_groups'] # 3)
        self.eval_num_probs = kwargs['eval_num_probs'] # 6)
        assert self.num_groups >= self.eval_num_groups
        self.debug_mode = kwargs.get('debug_mode', False)
        

        # self.seed_planner = self.mazelist_config.get('seed_planner', None)
        self.load_mazeEnv = kwargs.get('load_mazeEnv', True)
        self.world_dim = 2


        
        # ---------- 4. init parent class ---------
        multiproc_mode = kwargs.get('multiproc_mode', False)
        if not multiproc_mode:
            # create the env
            self.create_env()
            self.robot_0 = robot_0 = self.model_list[0].robot
            OfflineEnv.__init__(self, **kwargs)
            PybulletEnv.__init__(self, obs_low=robot_0.limits_low, obs_high=robot_0.limits_high,
                                action_low=np.zeros_like(robot_0.joints_max_force, dtype=np.float32),
                                action_high=robot_0.joints_max_force)
        else:
            self.model_list = [None for _ in range(self.num_groups)]
            OfflineEnv.__init__(self, **kwargs)
            dummy = np.zeros(shape=(7,), dtype=np.float32)
            PybulletEnv.__init__(self, obs_low=dummy, obs_high=dummy+1,
                                action_low=dummy,
                                action_high=dummy+1)
        
        self.env_type = 'static_maze2d'
        # ------- 5. Do some Checking ------

    def setup_wgrp(self):
        tmp_ng = self.num_groups if not self.is_eval else self.kwargs['eval_num_groups']
        self.rand_rec_group = RandRectangleGroup # could be a composed group

        self.recs_grp = self.rand_rec_group(
            env_list_name=self.env_name,
            num_groups=tmp_ng,
            num_walls=self.num_walls,
            maze_size=self.maze_size,
            half_extents=self.hExt_range,
            gap_betw_wall=self.gap_betw_wall,
            seed=self.rng_seed, # == self.rng_seed
            gen_data=self.gen_data,

            robot_class=self.robot_env.robot_class,
            is_eval=self.is_eval,
            rand_iter_limit=self.mazelist_config.get('rand_iter_limit', 1e5)
        )

        # (ng, n_v, 2), e.g. (5000, 6, 2), not to big
        # hExt could be different in composing cases
        self.wallLoc_list = np.copy(self.recs_grp.rec_loc_grp_list)
        self.hExt_list = np.copy(self.recs_grp.rec_hExt_grp_list)





    def create_env(self):
        '''create all envs that are in self.dv_group_list'''
        assert not hasattr(self, 'model_list')
        self.model_list = [None,] * self.num_groups
        for i_g in range(self.num_groups):

            if self.load_mazeEnv or i_g == 0:
                self.create_single_env(i_g)

            # break for eval mode
            if self.is_eval and i_g == (self.eval_num_groups - 1):
                break

    def create_single_env(self, env_idx) -> RandMaze2DEnv:
        '''put the created env to model_list, also return the same env instance'''
        if self.model_list[env_idx] is None:
            wall_locations = self.wallLoc_list[env_idx]
            wall_hExts = self.hExt_list[env_idx]

            ## automatic pass in planner_timeout, seed_planner
            env = self.robot_env(wall_locations, wall_hExts, self.robot_config,
                                     renderer_config={},  **self.mazelist_config)
            self.model_list[env_idx] = env

        # unload prev env
        # if env_idx > 0 and self.model_list[env_idx-1] is not None:
            # self.model_list[env_idx-1].unload_env()
        return self.model_list[env_idx]


    def create_mazeEnv_by_idx(self, env_idx):
        return self.create_single_env(env_idx)
    

    def create_env_by_pos(self, env_idx, wall_locations, wall_hExts) -> RandMaze2DEnv:
        '''set the env_idx to the specific user given env'''
        assert wall_locations.ndim == 2 and wall_hExts.ndim == 2 and isinstance(wall_locations, np.ndarray)
        
        env = self.robot_env(wall_locations, wall_hExts, self.robot_config,
                                     renderer_config={},  **self.mazelist_config)
        self.model_list[env_idx] = env
        return self.model_list[env_idx]



    def __getitem__(self, idx):
        '''return a dv group obj'''
        raise NotImplementedError()
        return self.dv_group_list[idx]

    

    def load2pybullet(self):
        '''seems to be useless'''
        raise NotImplementedError('the env_list should not be directly loaded')
    

    def get_env_name(self):
        hdf5_path = os.path.basename(self.dataset_url)
        env_name = hdf5_path.replace('.hdf5', '')
        print(f'get_env_name: {env_name}')
        return env_name


    def render_mazelist(self, savepath: str, paths, maze_idx_list: np.ndarray):
        '''
        render a large images given a list of paths and maze_idx
        maze_idx_list: (n, 1)
        '''

        renderer = self.model_list[0].renderer
        ms_list = self.maze_size[None,].repeat( len(paths), axis=0 )
        cpgrp_list = self.wallLoc_list[ maze_idx_list.flatten() ]
        hExtgrp_list = self.hExt_list[ maze_idx_list.flatten() ]

        renderer.composite(
            savepath, paths, 
                  ms_list, cpgrp_list, hExtgrp_list,
        )

