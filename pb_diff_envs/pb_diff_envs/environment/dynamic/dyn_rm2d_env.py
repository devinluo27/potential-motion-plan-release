import numpy as np
from pb_diff_envs.robot.point2d_robot import Point2DRobot
from pb_diff_envs.environment.offline_env import OfflineEnv
from pb_diff_envs.environment.pybullet_env import PybulletEnv
from pb_diff_envs.environment.abs_maze2d_env import AbsDynamicMaze2DEnv
from pb_diff_envs.environment.dynamic_rrgroup import DynamicRectangleWall, DynamicRecWallGroup
from pb_diff_envs.objects.trajectory import WaypointLinearTrajectory
from typing import List
from gym.utils import EzPickle
from colorama import Fore
import time
from pb_diff_envs.utils.maze2d_utils import get_is_collision_wtrajs
from pb_diff_envs.environment.dynamic.dyn_wtraj_gen import DynWallTrajGenerator
from pb_diff_envs.environment.dynamic.dyn_rm2d_renderer import DynamicRecRenderer
from pb_diff_envs.planner.sipp_planner import SippPlanner
import pdb

class DynamicRandMaze2DEnv(AbsDynamicMaze2DEnv, OfflineEnv, PybulletEnv, EzPickle):
    robot_class = Point2DRobot

    def __init__(self, wall_locations, wall_hExts, 
                 len_full_wtraj: int, 
                 num_sample_pts: int,
                 gap_betw_wall: int,
                robot_config, wp_seed, speed, renderer_config={}, **kwargs):
        '''

        wall_locations: only the start location, keep changing
        current wall_locations is in the DynamicRecWallGroup
        num_sample_pts: num of wtraj rand points before interp, include start end
        '''
        ## set hyper-param
        self.maze_size: np.ndarray = robot_config['maze_size']

        # self.o_env.get_obstacles()[:, :3] # np (20, 2)
        self.init_wlocs = wall_locations # center, keep changing
        self.wtraj_start_wlocs = np.copy(wall_locations)

        self.wall_hExts = wall_hExts
        self.num_walls = len(wall_locations)
        self.len_full_wtraj = len_full_wtraj
        self.num_sample_pts = num_sample_pts
        self.gap_betw_wall = gap_betw_wall

        self.speed = speed
        assert self.num_walls <= 8 and wall_hExts.ndim == 2
        self.dw_traj_gen = DynWallTrajGenerator(self.maze_size, 
                                                len_full_wtraj, num_sample_pts, 
                                                gap_betw_wall, speed)
        ## [Useless] this seed is expected to control the sequence of wtraj
        ## we need to fix the seed for goal for reproductivity
        self.rng_wtraj = np.random.default_rng(seed=wp_seed)
        self.total_cnt_wtraj = 0 # how many times we get a new wtraj
        self.buffer_wtraj = [None,] * 10 # keep track of latest ten wtrajs
        # self.update_dyn_wallgroup()


        # ---- continued, ** recWall_grp must be updated after every episode **
        self.robot_config = robot_config
        self.robot = Point2DRobot(recWall_grp=None, **robot_config)
        self.update_dyn_wallgroup()
        
        ## renderer for dynamic env
        self.renderer = DynamicRecRenderer(**renderer_config)


        assert type(self.robot) == self.robot_class, 'seems to be useless'

        self.min_to_wall_dist = robot_config['min_to_wall_dist']
        assert self.maze_size[0] - self.maze_size[1] < 1e-5, 'ensure square maze'
        assert self.init_wlocs.ndim == 2 and self.wall_hExts.ndim == 2


        PybulletEnv.__init__(self, obs_low=self.robot.limits_low, obs_high=self.robot.limits_high,
                             action_low=np.zeros_like(self.robot.limits_low, dtype=np.float32),
                             action_high=np.ones_like(self.robot.limits_low, dtype=np.float32))
        OfflineEnv.__init__(self, **kwargs)
        ## add support for picklize the object
        EzPickle.__init__(self, wall_locations, wall_hExts, 
                          robot_config, renderer_config={}, **kwargs)
        

        seed_planner = kwargs.get('seed_planner', None)
        assert seed_planner is None, 'seed planner will not affect performance'
        planner_num_batch = kwargs.get('planner_num_batch', 1000)
        k_nb = kwargs.get('k_nb', 50)
        print(Fore.CYAN + f'dyn env pnb: {planner_num_batch}, k_nb: {k_nb}, ', end='')
        print(f'rc: {robot_config}' + Fore.RESET)

        self.planner = SippPlanner(num_samples=planner_num_batch, 
                                   stop_when_success=True, k_neighbors=k_nb)
        self.planner_timeout = kwargs.get('planner_timeout', 200) # seconds
        self.planner_val = self.planner


        # self.wall_locations = self.o_env.get_obstacles()[:, :3] # np (20, 3)
        # print(f'self.o_env {self.o_env} {id(self.o_env)} {self.o_env.robot}')
        self.min_episode_distance = kwargs.get('min_episode_distance', self.maze_size[0] / 2 * 1.4)
        self.sample_cnt_limit = 1e4
        self.env_id = 'dynamic_maze2d'


        





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
        # wtrajs = self.dw_traj_gen.get_wtrajs_same_stepsize(wtrajs=wtrajs, speed=0.1)

        self.cur_wtrajs = wtrajs # to be checked
        print(f'[dyn env wtrajs] {wtrajs.shape}')
        recWall_list = []
        ## 2. inject wtraj to grp
        for i_w in range(self.num_walls):
            wpl_traj = WaypointLinearTrajectory(wtrajs[i_w])
            tmp = DynamicRectangleWall(self.wtraj_start_wlocs[i_w], 
                                       self.wall_hExts[i_w], wpl_traj)
            recWall_list.append(tmp)

        self.recWall_grp = DynamicRecWallGroup(recWall_list)
        ## For SIPP, abstraction
        self.objects = self.recWall_grp.dyn_recWall_list

        self.robot.recWall_grp = self.recWall_grp
        # self.robot = Point2DRobot(recWall_grp=self.recWall_grp, **self.robot_config)

        # 3. 
        # we assume that we must move the wall to the dest_wlocs before sample another

        self.buffer_wtraj[ self.total_cnt_wtraj % 10] = wtrajs
        self.total_cnt_wtraj += 1




    def get_rand_wtrajs(self):
        '''return a list of np2d (,2) '''
        if hasattr(self, 'wtraj_dest_wlocs'):
            assert (self.wtraj_start_wlocs == self.wtraj_dest_wlocs).all(), 'not sure if need, slow'

        wtrajs = self.dw_traj_gen.get_rand_linear_wtrajs(self.wtraj_start_wlocs, 
                                                         self.wall_hExts, self.rng_wtraj, 'last')

        return wtrajs



    def is_new_goal_valid(self, new_goal):
        '''check if the given goal location is valid'''
        # not in w end pos
        return self.robot.no_collision()
         
        


    def sample_1_episode(self, prev_pos):
        '''
        In this method, we need to change the robot pos to detect collision.
        Args:
        prev_pos (np): (2,) array of start pos
        '''
        assert (self._get_joint_pose() - prev_pos).sum() < 1e-5, f'in prev pose {self._get_joint_pose()}, {prev_pos}'
        assert self.recWall_grp.is_walls_in_start_pos(), ''
        assert self.robot.no_collision()

        ## ----- set to end pose, to check is goal valid -----
        self.recWall_grp.set_walls_at_time(-1)

        num_planner_fail = 0
        high = self.robot.limits_high # (14,)
        low = self.robot.limits_low # (14,)
        pos_reset = False
        cnt_iter = 0

        while True:
            cnt_iter += 1
            new_goal = np.random.uniform(low=low, high=high, size=(len(low),))


            if not self.state_fp(new_goal, -1):
                continue

            if num_planner_fail > 4:
                '''when fail times > 10, when set the starting pos to 0'''
                pos_reset = True
                # prev_pos = self.get_robot_free_pose() # reset 
                prev_pos = self.resample_robot_start_pos()
                print(Fore.RED + f'[Process Fail] num_planner_fail: {num_planner_fail}' + Fore.RESET)

            self.robot.set_config(new_goal)
            # pdb.set_trace()

            if np.abs(new_goal - prev_pos).sum() < self.min_episode_distance \
                and num_planner_fail <= 5 \
                and cnt_iter <= self.sample_cnt_limit: # 1000
                continue
            elif self.is_new_goal_valid(new_goal):
                # print(f'self.o_env {self.o_env} {id(self.o_env)}')
                # print('new_goal:', np.round(new_goal, 3))
                ## set to start pose
                self.robot.set_config(prev_pos)
                self.recWall_grp.set_walls_at_time(0)
                assert self.recWall_grp.is_walls_in_start_pos(), ''
                print('prev_pos', prev_pos, 'new_goal', new_goal)


                ## assume a solution exist!
                start_time = time.time()

                result_tmp = self.planner.plan(self, prev_pos, new_goal, timeout=('time', self.planner_timeout))


                if result_tmp.solution is not None:
                    elapsed_time = time.time() - start_time
                    

                    r_traj = np.array(result_tmp.solution) # list to array
                    episode_wtrajs = self.algin_w_to_r_traj_len(r_traj, np.copy(self.cur_wtrajs))

                    is_cols = get_is_collision_wtrajs(r_traj, episode_wtrajs, self.wall_hExts, self.min_to_wall_dist)
                    has_col = is_cols.sum() > 0


                    print(f'cur wall pose: {self.recWall_grp.get_walls_pos()}')
                    print(f'wtraj end pose: {self.recWall_grp.get_walls_pos_at_t(-1)}')

                    
                    if has_col:
                        ## ** we do not know where is the wall now! **
                        print(Fore.RED + f'Bad solution ncf:{is_cols.sum()} in {elapsed_time}s,')

                        print(f'wtrajs: {self.recWall_grp.get_wtrajs()}')
                        print(f'cur wall pose: {self.recWall_grp.get_walls_pos()}')
                        print(f'wtraj end pose: {self.recWall_grp.get_walls_pos_at_t(-1)}')

                        print(f'resample a target. # {num_planner_fail}' + Fore.RESET)
                        num_planner_fail += 1

                        self.recWall_grp.set_walls_at_time(-1) ## set it to final place for goal sampling
                        
                    else:
                        print(f'Solution is found in {elapsed_time}s.')
                        break

                else:
                    elapsed_time = time.time() - start_time
                    print(f'Failed to find a solution in {elapsed_time}s, resample a target. # {num_planner_fail}')
                    num_planner_fail += 1

                    ## if fail, also set back
                    self.recWall_grp.set_walls_at_time(-1) ## set it to final place for goal sampling


        print(f's1e: cnt {cnt_iter}, diff: {np.abs(new_goal - prev_pos).sum()}, len(sol): {len(result_tmp.solution)}')




        # ------------ Dynamic Custom -------------
        self.robot.set_config(new_goal)

        # assert ( episode_wtrajs == self.recWall_grp.get_wtrajs() ).all()

        # episode_wtrajs should be set
        # update wall dest
        # we assume that we must move the wall to the dest_wlocs before sample another
        self.wtraj_dest_wlocs = np.copy( episode_wtrajs[:, -1, :] ) # (n_w, 2)

        # -------- generate a new dynw grp with new wtraj -----
        self.wtraj_start_wlocs = self.wtraj_dest_wlocs
        ## where is the wall now?
        # print('get_walls_pos 1', self.recWall_grp.get_walls_pos()) # 1 2 same
        self.recWall_grp.set_walls_at_time(-1) #
        # print('get_walls_pos 2', self.recWall_grp.get_walls_pos())

        # get next wtrajs starting from the last w pose
        self.update_dyn_wallgroup()
        
        # ------------------------



        # result_tmp.solution, a numpy array contains start, waypoints, end
        return r_traj, {'pos_reset': pos_reset, 'wtrajs': episode_wtrajs, 
                        'pl_time': result_tmp.running_time,
                        'n_colchk': result_tmp.num_collision_check}


    
    def set_dyn_wallgroup(self, wtrajs):
        '''
        set wall group, given wtrajs, (nw, h, 2)
        '''
        # self.cur_wtrajs = wtrajs # to be checked
        print(f'[dyn env wtrajs] {wtrajs.shape}')
        recWall_list = []
        ## 2. inject wtraj to grp
        for i_w in range(self.num_walls):
            wpl_traj = WaypointLinearTrajectory(wtrajs[i_w])
            tmp = DynamicRectangleWall(wtrajs[i_w][0], 
                                       self.wall_hExts[i_w], wpl_traj)
            recWall_list.append(tmp)

        self.recWall_grp = DynamicRecWallGroup(recWall_list)
        ## For SIPP, abstraction
        self.objects = self.recWall_grp.dyn_recWall_list
        self.robot.recWall_grp = self.recWall_grp

        # 3. 
        # we assume that we must move the wall to the dest_wlocs before sample another

        # self.buffer_wtraj[ self.total_cnt_wtraj % 10] = wtrajs
        # self.total_cnt_wtraj += 1





    def resample_robot_start_pos(self):
        self.recWall_grp.set_walls_at_time(0)
        pose = self.get_robot_free_pose()
        self.recWall_grp.set_walls_at_time(-1) # for valid goal check
        return pose



    ## ------------- rendering ----------------
    def render_1_traj(self, r_traj: np.ndarray, wtrajs: np.ndarray, 
                      img_type='gif', savepath=None):
        '''
        wtrajs: shape 1. (h, nw, dim) 2. (nw, h, dim) 
        Renderer needs: 
        Return a gif or image of the env
        '''
        # wtrajs = wtrajs.append()
        if img_type == 'gif':
            return self.renderer.render_gif_v2(savepath, self.maze_size, r_traj, 
                                            wtrajs, self.wall_hExts)
        elif img_type == 'png':
            return self.renderer.render_png_v2(savepath, self.maze_size, r_traj, 
                                            wtrajs, self.wall_hExts)
        else:
            raise NotImplementedError()

    def render(self):
        wtrajs = self.recWall_grp.get_wtrajs()
        r_traj = np.zeros_like(wtrajs[0])
        return self.renderer.render_png_v2(None, self.maze_size, r_traj, 
                                            wtrajs, self.wall_hExts)
    

    def algin_w_to_r_traj_len(self, r_traj, wtrajs):
        '''
        returns: 
        wtraj: np (nw, time, 2)
        '''
        residual = len(r_traj) - wtrajs.shape[1]

        ## 2. ------------ design choice ---------------
        ## A. we pad the wall traj to be equal to robot traj
        ## B. we trim the redundant wall traj to decrease to len of robot traj

        if residual > 0: # robot longer
            pad = wtrajs[:, -1:, :].repeat(residual, axis=1) # (n_w, 1, 2) -> (n_w, res, 2)
            wtrajs = np.append( wtrajs, pad, axis=1 )
        elif residual < 0: # wall longer
            print(Fore.RED + f'[ wtraj is longer, residual < 0 ]' + Fore.RESET)
            wtrajs = wtrajs[:, :len(r_traj), :]
        
        assert wtrajs.shape[1] == r_traj.shape[0]

        return wtrajs







