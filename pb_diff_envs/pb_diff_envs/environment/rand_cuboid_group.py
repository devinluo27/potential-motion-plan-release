import pybullet as p
import numpy as np
from pb_diff_envs.environment.abstract_env import AbstractEnv, load_table
from pb_diff_envs.utils.kuka_utils_luo import from_rel2abs_path
from pb_diff_envs.robot.multi_robot.dual_kuka_robot import DualKukaRobot
from pb_diff_envs.robot.kuka_robot import KukaRobot
import pdb, time, os
from pb_diff_envs.utils.save_utils import save_scatter_fig
from colorama import Fore
from tqdm import tqdm

class RandCuboidGroup(AbstractEnv):
    def __init__(self, env_list_name:str, num_groups, num_voxels, 
                 start_end, void_range, half_extents, seed, gen_data,
                 robot_class, is_eval=False,
                 **kwargs) -> None:
        '''
        start_end, void_range, half_extents (all np)
        start_end: the start (neg) and end (positive) for *cuboid's center location*
        void_range: different from VoxelGroup, the void_range means *nothing should be in the given range*
        half_extents: should be a fix size
        '''
        
        self.rng = np.random.default_rng(seed)
        self.rng_color = np.random.default_rng(100)


        self.env_list_name = env_list_name
        self.num_groups = num_groups
        self.num_voxels = num_voxels

        self.start_end = np.copy(start_end) # convert to numpy if is list (3,2)
        self.void_range = np.copy(void_range) # np (3, 2)
        self.half_extents = np.copy(half_extents) # tuple of size 3 -> np
        assert half_extents.shape == (3,), 'only support fix size now'
        assert start_end[2, 0] >= half_extents[2], 'cube cannot touch tables'
        
        self.voxel_size = half_extents * 2 # tuple of size 3
        self.base_orn = [0, 0, 0, 1]
        self.sort_wloc = True # sort the order as other env does

        
        self.check_range_symmetric(start_end)
        self.check_range_symmetric(void_range)
        self.get_default_valid_range() ## just to print

        self.gen_data = gen_data # if False, directly load
        self.GUI = kwargs.get('GUI', False)
        self.debug_mode = kwargs.get('debug_mode', False)
        self.robot_class = robot_class
        self.load_vis_objects = True # if load robot and table
        self.is_eval = is_eval
        self.rand_iter_limit = kwargs['rand_iter_limit'] # default 1e5
        self.gap_betw_wall = kwargs.get('gap_betw_wall', None)

        if self.gen_data:
            self.create_envs()
        else:
            # ng, n_c, 3
            self.voxel_grp_list = self.load_voxel_grp_list()
        
        # setup the hExts
        self.voxel_hExt_grp_list = np.tile( self.half_extents, (self.num_groups, self.num_voxels, 1) )
        assert (self.voxel_grp_list.shape == self.voxel_hExt_grp_list.shape)
        

    
    def create_envs(self):
        '''set voxel_grp_list'''
        self.load()

        voxel_grp_list: list[np.ndarray] = [] # a list of np2d: n_c,3
        for i in tqdm(range(self.num_groups)):
            cub_grp, _ = self.sample_one_valid_env()
            voxel_grp_list.append(cub_grp)
        
        # np3d: ng, n_cube, 3
        voxel_grp_list = np.array(voxel_grp_list)

        self.unload()
        self.voxel_grp_list = voxel_grp_list
        self.save_vg_list()
        return
            

    def sample_one_valid_env(self) -> np.ndarray:
        '''create one env config without any overlap
        returns:
        array (n_c,3) of 3d position '''
        self.reset()
        # self.set_default_shape()
        pos_list = []
        hExt_list = []
        cnt_iter = 0
        exist_mb_ids = []
        while len(pos_list) < self.num_voxels:
            center_pos, hExt = self.sample_xyz_hExt()
            ## make sure hExt takes effect here, not impl yet
            mb_id = self.create_multibody(center_pos, hExt)

            # print(p.getBodyInfo(tmp_id))
            p.performCollisionDetection()
            c_pts = p.getContactPoints()
            # print(f'c_pts: {c_pts}', ) # mass must be > 0
            has_collision = len(c_pts) > 0
            gap_valid =  self.check_gap_betw_wall(mb_id, exist_mb_ids)
                
            if has_collision or (not gap_valid):
                self.remove_multibody(mb_id)
            else:
                pos_list.append(center_pos.tolist()) # for sorting
                hExt_list.append(hExt)
                exist_mb_ids.append(mb_id)

            cnt_iter += 1
            if cnt_iter > self.rand_iter_limit: # deadlock, reset everything
                print(pos_list, exist_mb_ids)
                self.reset()
                pos_list = []
                hExt_list = []
                exist_mb_ids = []
                cnt_iter = 0
            # if cnt_iter % 10000 == 0: # 5s
                # print('cnt_iter', cnt_iter)

        # print('cnt_iter', cnt_iter) # usually < 1000
        self.reset()
        if self.sort_wloc:
            ## pos_list must be 2d, a list of list
            idx_and_pos = sorted(enumerate(pos_list), key=lambda i:i[1]) # list of a tuple (idx, w)
            pos_list = [i_p[1] for i_p in idx_and_pos] # i_p a tuple, still a list of list
            pos_sort_idx = [i_p[0] for i_p in idx_and_pos] # a list of int
            
            hExt_list = np.array(hExt_list)[pos_sort_idx] # switch order correspondingly
        else:
            assert False
            hExt_list = np.array(hExt_list)

        pos_list = np.array(pos_list) # n_c, 3
        return pos_list, hExt_list

    def check_gap_betw_wall(self, new_mb_id, exist_mb_ids):
        '''return True is nothing is around, so valid'''
        if self.gap_betw_wall is None or (not self.gap_betw_wall):
            return True
        for e_id in exist_mb_ids:
            '''closest points compute surface to surface distance, not center to center'''
            tmp_pts = p.getClosestPoints(new_mb_id, e_id, distance=self.gap_betw_wall)
            if len(tmp_pts) > 0:
                return False
        return True
    
    def sample_xyz_hExt(self) -> np.ndarray: # 1d
        '''
        sample a (x y z) 3d location and half_extent,
        making sure that it avoids forbidden area
        Returns:
        two np 1d (3,)
        '''
        # we can randomly sample something here
        hExt = self.half_extents.copy()
        while True:
            center_pos = self.rng.uniform(low=self.start_end[:, 0], high=self.start_end[:, 1])
            if self.is_in_void_range(center_pos, half_extents=hExt):
                continue
            else:
                break

        return center_pos, hExt


    def is_in_void_range(self, pos, half_extents):
        '''
        NOTE we only check X, Y now
        args:
        a xyz location:
        half_extents of the cuboid:'''
        vr = self.get_center_void_range(half_extents)
        ## 1. check xy
        xy_1 = np.all(pos[:2] > vr[:2, 0]) # greater than neg
        xy_2 = np.all(pos[:2] < vr[:2, 1])

        ## 2. check z
        ## not implement

        return xy_1 and xy_2


    def get_center_void_range(self, half_extents):
        ''' 
        self.void_range is forbidden area of any part of a cuboid
        this func compute forbidden area of the center of cuboid'''

        vr = np.copy(self.void_range)
        ## 1. set xy
        vr[:2, 0] = vr[:2, 0] - half_extents[:2] # neg (2,)
        vr[:2, 1] = vr[:2, 1] + half_extents[:2] # 

        ## 2. set z, not implemented, no need
        
        # print('center vr', vr)
        return vr


    def set_default_shape(self):
        self.v_id_default = p.createVisualShape(p.GEOM_BOX, halfExtents=self.half_extents)
        self.c_id_default = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extents)
    


    def create_multibody(self, base_pos, hExt):
        '''create a cube multibody'''
        assert (hExt == self.half_extents).all()
        # v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=hExt)
        c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=hExt)

        mb_id = p.createMultiBody(baseMass=0.1,
                                     baseCollisionShapeIndex=c_id,
                                     baseVisualShapeIndex=-1, # v_id,
                                     basePosition=base_pos,
                                     baseOrientation=self.base_orn,
                                    #  linkJointTypes=[p.JOINT_FIXED,],
                                     )
        
        color = self.rng_color.uniform(0, 1, size=(3,)).tolist() + [1]
        p.changeVisualShape(mb_id, linkIndex=-1, rgbaColor=color)
        # if self.debug_mode:
            # pdb.set_trace()
        return mb_id
    
    def get_shape(self):
        pass
    
    def get_rand_shape(self):
        pass


    # --------- pybullet interface -----------
    def remove_multibody(self, mb_id):
        p.removeBody(mb_id)


    def load(self):
        self.initialize_pybullet(GUI=self.GUI)
        p.setGravity(0., 0., 0.) # a must, otherwise voxel will fall

    def unload(self):
        self.unload_env()
    
    def reset(self):
        p.resetSimulation() 
        if self.load_vis_objects:
            self.load_table()
            self.load_robot()

    # load other objects 
    def load_table(self):
        table = from_rel2abs_path(__file__, './../data/robot/kuka_iiwa/table_dualkuka14d/table.urdf')
        self.table_id = load_table(table)

    def load_robot(self):
        robot = self.robot_class()
        item_ids = robot.load() # list

    # ---------------------------------------


    # --------- save and load the wall locations -----------
    def get_npyname(self):
        self.prefix = from_rel2abs_path(__file__, '../datasets/rand_vgrp/')
        os.makedirs(self.prefix, exist_ok=True)
        if self.is_eval:
            npyname = f'{self.prefix}/{self.env_list_name}_eval.npy'
        else:
            npyname = f'{self.prefix}/{self.env_list_name}.npy'
        return npyname
    
    ## np (num_groups,3)
    def save_vg_list(self):
        assert self.voxel_grp_list.shape[0] == self.num_groups
        npyname = self.get_npyname()

        ## check instead
        if os.path.exists(npyname) and 'testOnly' not in npyname:
            self.check_matched()
        else:
            np.save(npyname, self.voxel_grp_list)
            if 'testOnly' not in npyname:
                os.chmod(npyname, 0o444)

    def load_voxel_grp_list(self):
        voxel_grp_list = np.load(self.get_npyname())
        return voxel_grp_list

    def check_matched(self):
        # assert not self.gen_data, 'gen_data will perturb seed state'
        assert (self.voxel_grp_list == self.load_voxel_grp_list()).all(), 'Please delete the corresponding npy file if already exists'

    # ------------------------------------------------------


    # ------------------ visualization ---------------------

    def get_default_valid_range(self):
        '''print the default valid range with fix hExt'''
        center_void = self.get_center_void_range(self.half_extents)
        print(f'center_void {center_void}')
        print(Fore.RED + f'[RandCuboidGroup] nv {self.num_voxels}, available location range:')
        xy = ['x', 'y']
        for i in range(2):
            neg = ( self.start_end[i, 0], center_void[i, 0] )
            pos = ( center_void[i, 1], self.start_end[i, 1] )
            print(f'{xy[i]} neg: {neg} pos: {pos}' )
        print('no constraints on z for now.'+Fore.RESET)



    def simulate_voxel_grp(self, grp_idx):
        # assert self.debug_mode
        self.load()
        self.load_cuboids(grp_idx)
        self.set_camera_angle()
        while True:
            p.stepSimulation()
            time.sleep(0.1)

    def load_cuboids(self, grp_idx):
        '''used in visualization, load cuboids of grp_idx one by one'''
        assert p.isConnected()
        self.reset()
        cub_grp = self.voxel_grp_list[grp_idx]
        for i in range(len(cub_grp)):
            print(f'cub_grp[{i}]:', cub_grp[i])
            self.create_multibody(cub_grp[i], self.half_extents)

    def set_camera_angle(self):
        camera_distance = 3.2
        camera_yaw = 50
        camera_pitch = -50
        camera_target_position = [0, 0, 0]
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target_position,
        )

    def vis_xy2d_sampling(self, opt=1):
        assert self.debug_mode
        prefix = from_rel2abs_path(__file__, '../datasets/rand_vgrp/')
        os.makedirs(prefix, exist_ok=True)
        fig_path = f'{prefix}/{self.env_list_name}_{opt}.png'
        points = []
        if opt == 1:
            # whole space
            for _ in range(2000):
                pt = np.random.uniform(low=self.start_end[:, 0], high=self.start_end[:, 1])
                points.append(pt[:2])
        elif opt == 2:
            raise NotImplementedError
        
        return save_scatter_fig(points, fig_path) # retrun plt.gcf()
    
    # ---------------------------------------------


    # ---------------- checking -------------------
    def check_range_symmetric(self, r):
        assert r.shape == (3, 2)
        r = np.abs(r)
        assert (r[:, 0] - r[:, 1] < 1e-6).all()
    
