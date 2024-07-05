import numpy as np
import pdb
import os
from pb_diff_envs.environment.static.comp_cuboid_group import ComposedRandCuboidGrp
from pb_diff_envs.environment.static.comp_kuka_env import ComposedRandKukaEnv, ComposedRandDualKuka14dEnv
from pb_diff_envs.environment.static.kuka_rand_vgrp import Kuka_VoxelRandGroupList
from pb_diff_envs.objects.static.voxel_group import DynVoxelGroup
from pb_diff_envs.environment.offline_env import OfflineEnv
from pb_diff_envs.environment.pybullet_env import PybulletEnv


class ComposedKuka_VoxelRandGroupList(Kuka_VoxelRandGroupList):
    def __init__(self, robot_env, num_groups, n_comp,
                 num_walls_c, 
                 start_end, void_range, orn_range,
                 hExt_range_c, comp_type, is_static=True, seed=270, **kwargs):
        
        assert robot_env in [ComposedRandKukaEnv, ComposedRandDualKuka14dEnv]

        self.n_comp = n_comp
        self.hExt_range_c = hExt_range_c
        self.num_walls_c = num_walls_c
        self.comp_type = comp_type
        self.grp_seed_diff = kwargs['mazelist_config']['grp_seed_diff']
        self.r_cfg = dict(num_walls_c=num_walls_c)

        # --------------------------------

        self.robot_env = robot_env # a class to call


        '''when eval, use a different seed to generate voxel loc!
        used when generating val problems
        '''
        ## 1. -------- setup hyper-param value -----------
        self.is_eval = kwargs.get('is_eval', False)
        self.eval_seed_offset = kwargs.get('eval_seed_offset', 0)
        if self.is_eval:
            assert self.eval_seed_offset != 0
            seed = seed + self.eval_seed_offset

        # self.rng = np.random.default_rng(seed) # the rng only to generate comb
        self.rng_color = np.random.default_rng(100)
        self.rng_seed = seed
        self.num_groups = num_groups
        self.num_voxels = num_walls_c.sum().item()
        self.mazelist_config = kwargs['mazelist_config']
        self.samples_per_env = kwargs['samples_per_env']

        self.dv_group_list = []
        self.colors = []

        self.start_end = np.copy(start_end) # convert to numpy if is list
        self.void_range = np.copy(void_range)
        hExt_range = hExt_range_c[0] # not good
        self.hExt_range = np.copy(hExt_range_c[0])
        assert start_end[2][0] >= hExt_range[2]
        assert start_end[2][0] - (start_end[2][0] / hExt_range[2]) * hExt_range[2] < 1e-7
        assert hExt_range.shape == (3,)

        # ----- setup config to init robot env ------
        if self.robot_env == ComposedRandDualKuka14dEnv:
            c_eps = 0.06
        elif self.robot_env == ComposedRandKukaEnv:
            c_eps = 0.04
        self.robot_config = dict(collision_eps=c_eps)
        self.world_dim = 3


        ## 2. ------ setup wall locations ----------
        self.dataset_url = kwargs['dataset_url']
        self.env_name = self.get_env_name()
        # only used when generate the config the first time; otherwise, simply load
        self.gen_data = kwargs.get('gen_data', False)
        tmp_ng = num_groups if not self.is_eval else kwargs['eval_num_groups'] # checkparam

        self.mazelist_config['rand_iter_limit'] = self.mazelist_config.get('rand_iter_limit', 1e5)

        self.cuboid_grp = ComposedRandCuboidGrp(
            env_list_name=self.env_name,
            num_groups=tmp_ng,
            n_comp=self.n_comp,
            num_walls_c=num_walls_c,
            start_end=start_end,
            void_range=void_range,
            half_extents_c=hExt_range_c,
            seed=seed,
            gen_data=self.gen_data,
            GUI=False,
            ## NOTE
            robot_class=robot_env.robot_class,
            is_eval=self.is_eval,
            **self.mazelist_config,
        )
        # checkparam (ng, n_v, 3), e.g. (5000, 6, 3), not to big
        # self.wallLoc_list = np.copy(self.cuboid_grp.voxel_grp_list)

        # (ng, n_v, 2), e.g. (5000, 6, 2), not to big
        # hExt could be different in composing cases
        self.recs_grp = self.cuboid_grp
        self.wallLoc_list = self.recs_grp.wallLoc_list.copy() # [ng, nw1+nw2, 2]
        self.hExt_list = self.recs_grp.hExt_list.copy() # [ng, nw1+nw2, 2]
        ## 

        # -------- setup some more values ----------
        # split to how many parts when generating dataset
        self.gen_num_parts = kwargs['gen_num_parts']
        self.no_duplicate_maze = self.mazelist_config.get('no_duplicate_maze', False)
        assert not self.no_duplicate_maze
        # if True, load trained env and exclude them; 
        # if False, simply generate env config with another seed
        self.eval_num_groups = kwargs['eval_num_groups'] # 3) checkparam
        self.train_eval_no_overlap = kwargs.get('train_eval_no_overlap', False)
        assert not self.train_eval_no_overlap
        self.eval_num_probs = kwargs['eval_num_probs'] # 6)
        self.debug_mode = kwargs.get('debug_mode', False)
        

        # self.seed_planner = self.mazelist_config.get('seed_planner', None)
        assert start_end.shape == (3,2) and void_range.shape == (3,2), '3 axis'
        self.load_mazeEnv = kwargs.get('load_mazeEnv', True)
        if not orn_range:
            # should be 4 * 2
            self.orn_range = [(0,0,0,1),] # make sure both are in canonical pose
            assert (np.array(self.cuboid_grp.base_orn) == np.array(self.orn_range)).all()
        else:
            raise NotImplementedError()
        # print(f'void_range shape:', void_range.shape)

        # 4. -------- setup a list of DynVoxelGroup ------------
        # ------- be careful of hExt and orn in the future -----
        # generate voxel group
        for i_g in range(self.num_groups):
            xyz_list, colors = self.get_rand_xyz(i_g) # np, list
            # print(f'{i_g} xyz_list', xyz_list) # (20, 3)
            orn_list = self.orn_range * self.num_voxels # list
            
            hExt_list = [hExt_range_c[0]] * self.num_voxels # list
            # print(xyz_list.shape, colors[0], len(colors)) # (20, 3)
            dvg = DynVoxelGroup(xyz_list, orn_list, hExt_list, colors, is_static)
            self.dv_group_list.append(dvg)

            ## in eval, break at time
            if self.is_eval and i_g == (self.eval_num_groups - 1):
                break

        self.maze_arr_list = self.wallLoc_list # only a place holder for eval code


        
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
        
        
        # ------- 5. Do some Checking ------


        if hasattr(self, 'num_walls'):
            del self.num_walls
        del self.hExt_range



class ComposedDualKuka_VoxelRandGroupList(ComposedKuka_VoxelRandGroupList):
    pass