from pb_diff_envs.objects.abstract_object import AbstractObject
import pybullet as p
import numpy as np
import pdb
from typing import Union
from pb_diff_envs.environment.offline_env import OfflineEnv
from pb_diff_envs.environment.static.rand_dualkuka14d_env import RandDualKuka14dEnv
from pb_diff_envs.environment.static.rand_kuka_env import RandKukaEnv
from pb_diff_envs.environment.pybullet_env import PybulletEnv
import math, os
from pb_diff_envs.objects.static.voxel_group import DynVoxelGroup
from pb_diff_envs.environment.rand_cuboid_group import RandCuboidGroup

class DualKuka_VoxelRandGroupList(OfflineEnv, PybulletEnv):

    def __init__(self, robot_env, num_groups, num_voxels, start_end: np.ndarray,
                 void_range, 
                 orn_range: Union[int, tuple],
                 hExt_range, is_static=True, seed=270, 
                 **kwargs):
        '''

        '''
        self.robot_env = robot_env # a class to call
        assert self.robot_env in [RandDualKuka14dEnv, RandKukaEnv,]

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
        self.num_voxels = num_voxels
        self.mazelist_config = kwargs['mazelist_config']
        self.samples_per_env = kwargs['samples_per_env']

        self.dv_group_list = []
        self.colors = []

        self.start_end = np.copy(start_end) # convert to numpy if is list
        self.void_range = np.copy(void_range)
        self.hExt_range = np.copy(hExt_range)
        assert start_end[2][0] >= hExt_range[2]
        assert start_end[2][0] - (start_end[2][0] / hExt_range[2]) * hExt_range[2] < 1e-7
        assert hExt_range.shape == (3,)

        # ----- setup config to init robot env ------
        if self.robot_env in [RandDualKuka14dEnv,]:
            c_eps = 0.06
        elif self.robot_env in [RandKukaEnv,]:
            c_eps = 0.04
        self.robot_config = dict(collision_eps=c_eps)
        self.world_dim = 3


        ## 2. ------ setup wall locations ----------
        self.dataset_url = kwargs['dataset_url']
        self.env_name = self.get_env_name()
        # only used when generate the config the first time; otherwise, simply load
        self.gen_data = kwargs.get('gen_data', False)
        tmp_ng = num_groups if not self.is_eval else kwargs['eval_num_groups'] # checkparam
        self.cuboid_grp = RandCuboidGroup(
            env_list_name=self.env_name,
            num_groups=tmp_ng,
            num_voxels=num_voxels,
            start_end=start_end,
            void_range=void_range,
            half_extents=hExt_range,
            seed=seed,
            gen_data=self.gen_data,
            GUI=False,
            ##
            robot_class=robot_env.robot_class,
            is_eval=self.is_eval,
            rand_iter_limit=self.mazelist_config.get('rand_iter_limit', 1e5),
            gap_betw_wall=self.mazelist_config.get('gap_betw_wall', None),
        )
        # (ng, n_v, 3), e.g. (5000, 6, 3), not to big
        self.wallLoc_list = np.copy(self.cuboid_grp.voxel_grp_list)


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
        # print(self.all_xyz_combs)

        # 4. -------- setup a list of DynVoxelGroup ------------
        # ------- be careful of hExt and orn in the future -----
        # generate voxel group
        for i_g in range(self.num_groups):
            xyz_list, colors = self.get_rand_xyz(i_g) # np, list
            # print(f'{i_g} xyz_list', xyz_list) # (20, 3)
            orn_list = self.orn_range * self.num_voxels # list
            
            # correct code, but do not need to prevent bugs
            # self.wallLoc_list.append(np.concatenate([xyz_list, orn_list], axis=1))
            hExt_list = [hExt_range] * num_voxels # list
            # print(xyz_list.shape, colors[0], len(colors)) # (20, 3)
            dvg = DynVoxelGroup(xyz_list, orn_list, hExt_list, colors, is_static)
            self.dv_group_list.append(dvg)
            # pdb.set_trace()
            ## in eval, break at time
            if self.is_eval and i_g == (self.eval_num_groups - 1):
                break

        self.maze_arr_list = self.wallLoc_list # only a place holder for eval code

        
        # ---------- 4. init parent class ---------
        multiproc_mode = kwargs.get('multiproc_mode', False)
        if not multiproc_mode:
            ## create the env
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
        
        
        # ------- 5. Do some Checking if necessary -------
        ## ...
        self.env_type = 'dualkuka14d'



    def create_env(self):
        '''create all envs that are in self.dv_group_list'''
        assert not hasattr(self, 'model_list')
        self.model_list = []
        for i in range(self.num_groups):
            objs = self.dv_group_list[i].dv_list # a list of DynamicVoxel objs
            if self.load_mazeEnv or i == 0:
                ## automatic pass in planner_timeout, seed_planner
                env = self.robot_env(objects=objs, robot_config=self.robot_config, **self.mazelist_config)
            else:
                env = None
            self.model_list.append(env)
            if self.is_eval and i == (self.eval_num_groups - 1):
                break

    def create_single_env(self, env_idx):
        '''used only after init'''
        if self.model_list[env_idx] is None:
            objs = self.dv_group_list[env_idx].dv_list # a list of DynamicVoxel objs
            ## automatic pass in planner_timeout, seed_planner
            env = self.robot_env(objects=objs, robot_config=self.robot_config, **self.mazelist_config)
            self.model_list[env_idx] = env
        # unload prev env
        if env_idx > 0 and self.model_list[env_idx-1] is not None:
            self.model_list[env_idx-1].unload_env()
        return self.model_list[env_idx]

    def create_mazeEnv_by_idx(self, env_idx):
        return self.create_single_env(env_idx)
    
    def create_env_by_pos(self, env_idx, xyz_list, wall_hExts=None):
        d_tmp = self.dv_group_list[env_idx]
        print('ori xyz_list:', d_tmp.xyz_list.shape, d_tmp.xyz_list)
        dvg = DynVoxelGroup(xyz_list, d_tmp.orn_list, d_tmp.hExt_list, d_tmp.colors, d_tmp.is_static)
        self.dv_group_list[env_idx] = dvg
        self.model_list[env_idx] = None
        self.wallLoc_list[env_idx] = xyz_list # might use this for inference
        return self.create_single_env(env_idx)



    def __getitem__(self, idx):
        '''return a dv group obj'''
        return self.dv_group_list[idx]


    def get_rand_xyz(self, i_g):
        '''simply load from given list'''
        wallLoc = self.wallLoc_list[i_g]
        assert wallLoc.shape == (self.num_voxels, 3)
        if hasattr(self, 'num_walls_c'): # composed
            c1 = np.array( [ [141, 153, 174] ] * self.num_walls_c[0] ) # 4, 3
            # c2 = np.array( [ [184, 192, 255] ] * self.num_walls_c[1] ) # 3, 3
            c2 = np.array( [ [255, 132, 0],] * self.num_walls_c[1] ) # 3, 3, [252, 163, 17] 
            colors = np.concatenate([c1, c2], axis=0) / 255.0 # 7,3
            colors = np.concatenate([colors, [[0.8,]]*self.num_voxels], axis=1).tolist()
        else:
            colors = self.rng_color.uniform(0, 1, size=(self.num_voxels, 3)) # color: same as below
            colors = np.concatenate([colors, [[1,]]*self.num_voxels], axis=1).tolist()

        # pdb.set_trace()
        return wallLoc, colors
    

    def check_duplicate_maze(self):
        '''very slow, can use for sanity check, so depricated'''
        # assert False
        for i_g in range(self.xyz_idx_list.shape[0]):
            xyz_idx = self.xyz_idx_list[i_g]
            is_eq = np.all( self.xyz_idx_list == xyz_idx, axis=1 ) # (num_groups,)
            appear_cnt = is_eq.sum() # is_found duplicate
            assert appear_cnt == 1
        print(f'[voxel_group] no duplicate maze')

        

    def load2pybullet(self):
        '''seems to be useless'''
        raise NotImplementedError('the env_list should not be directly loaded')
    
    def get_env_name(self):
        hdf5_path = os.path.basename(self.dataset_url)
        env_name = hdf5_path.replace('.hdf5', '')
        print(f'get_env_name: {env_name}')
        return env_name


