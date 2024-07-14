import numpy as np
from pb_diff_envs.robot.point2d_robot import Point2DRobot
from pb_diff_envs.environment.offline_env import OfflineEnv
from pb_diff_envs.environment.pybullet_env import PybulletEnv
from pb_diff_envs.environment.abs_maze2d_env import AbsStaticMaze2DEnv
from pb_diff_envs.planner.bit_star_planner import BITStarPlanner
from pb_diff_envs.environment.static.rand_maze2d_renderer import RandMaze2DRenderer
from pb_diff_envs.environment.rand_rec_group import RectangleWall, RectangleWallGroup
import imageio
from gym.utils import EzPickle
from colorama import Fore
import time

class RandMaze2DEnv(AbsStaticMaze2DEnv, OfflineEnv, PybulletEnv, EzPickle): # maybe custom an abstraction?
    robot_class = Point2DRobot

    def __init__(self, wall_locations, wall_hExts, robot_config, renderer_config={}, **kwargs):
        '''
        3 gaps: 
        a. gap between two walls to ensure connectivity
        b. gap between each wall and the four edges (same gap value as in a.)
        c. min gap between robot and wall
        we will pack wall info to a class and pass to the robot
        '''
        
        ## set hyper-param
        self.maze_size:np.ndarray = robot_config['maze_size']

        # self.o_env.get_obstacles()[:, :3] # np (20, 2)
        self.wall_locations = wall_locations # center
        self.wall_hExts = wall_hExts
        self.num_walls = len(self.wall_locations)
        self.create_wallgroup()


        self.robot = Point2DRobot(recWall_grp=self.recWall_grp, **robot_config)
        self.renderer = RandMaze2DRenderer(**renderer_config)
        assert type(self.robot) == self.robot_class, 'seems to be useless'

        self.min_to_wall_dist = robot_config['min_to_wall_dist']
        assert self.maze_size[0] - self.maze_size[1] < 1e-5, 'ensure square maze'
        assert self.wall_locations.ndim == 2 and self.wall_hExts.ndim == 2


        PybulletEnv.__init__(self, obs_low=self.robot.limits_low, obs_high=self.robot.limits_high,
                             action_low=np.zeros_like(self.robot.limits_low, dtype=np.float32),
                             action_high=np.ones_like(self.robot.limits_low, dtype=np.float32))
        OfflineEnv.__init__(self, **kwargs)
        ## add support for picklize the object
        EzPickle.__init__(self, wall_locations, wall_hExts, 
                          robot_config, renderer_config={}, **kwargs)
        

        seed_planner = kwargs.get('seed_planner', None)
        assert seed_planner is None, 'seed planner will not affect performance'
        planner_num_batch = kwargs.get('planner_num_batch', 400)
        self.planner = BITStarPlanner(num_batch=planner_num_batch, stop_when_success=True, seed=seed_planner)
        self.planner_timeout = kwargs.get('planner_timeout', 30) # seconds

        self.planner_val = self.planner

        # self.wall_locations = self.o_env.get_obstacles()[:, :3] # np (20, 3)
        # print(f'self.o_env {self.o_env} {id(self.o_env)} {self.o_env.robot}')

        ## ----- metric -----
        self.min_episode_distance = kwargs.get('min_episode_distance', self.maze_size[0] / 2 * 1.4)
        self.epi_dist_type = kwargs.get('epi_dist_type', 'sum')

        # print(Fore.BLUE + f'epd: {self.min_episode_distance}, {self.epi_dist_type}' + Fore.RESET)
        self.sample_cnt_limit = 1e5
        self.env_id = 'static_maze2d'
        self.kwargs = kwargs # from mazelist_config



    def create_wallgroup(self):
        '''init wall group for the collision checking'''
        recWall_list = []
        for i_w in range(self.num_walls):
            tmp = RectangleWall(self.wall_locations[i_w], self.wall_hExts[i_w])
            recWall_list.append(tmp)

        self.recWall_grp = RectangleWallGroup(recWall_list)

    


    def sample_1_episode(self, prev_pos):
        '''
        In this method, we need to change the robot pos to detect collision.
        Args:
        prev_pos (np): (2,) array of start pos
        '''
        assert (self._get_joint_pose() - prev_pos).sum() < 1e-5, 'in prev pose'
        assert self.robot.no_collision()
        num_planner_fail = 0
        high = self.robot.limits_high # (14,)
        low = self.robot.limits_low # (14,)
        pos_reset = False
        cnt_iter = 0

        while True:
            cnt_iter += 1
            new_goal = np.random.uniform(low=low, high=high, size=(len(low),))


            if num_planner_fail > 4:
                '''when fail times > 10, when set the starting pos to 0'''
                pos_reset = True
                prev_pos = self.get_robot_free_pose() # reset to 0 + noise
                print(Fore.RED + f'[Process Fail] num_planner_fail: {num_planner_fail}' + Fore.RESET)

            self.robot.set_config(new_goal)
            epi_dist = self.get_epi_dist(prev_pos, new_goal)

            if epi_dist < self.min_episode_distance \
                and num_planner_fail <= 5 \
                and cnt_iter <= self.sample_cnt_limit: # 1000
                continue
            elif self.robot.no_collision():
                # print(f'self.o_env {self.o_env} {id(self.o_env)}')
                # print('new_goal:', np.round(new_goal, 3))
                ## assume a solution exist!
                start_time = time.time()
                # result_tmp = self.planner.plan(self.o_env, prev_pos, new_goal, timeout=('time', self.planner_timeout))
                result_tmp = self.planner.plan(self, prev_pos, new_goal, timeout=('time', self.planner_timeout))


                if result_tmp.solution is not None:
                    elapsed_time = time.time() - start_time
                    # print(f'Solution is found in {elapsed_time}s.')
                    break
                else:
                    elapsed_time = time.time() - start_time
                    print(f'Failed to find a solution in {elapsed_time}s, resample a target. # {num_planner_fail}')
                    num_planner_fail += 1


        print(f'cnt {cnt_iter} diff: {np.abs(new_goal - prev_pos).sum()}, len(sol): {len(result_tmp.solution)}')
        # pdb.set_trace()
        self.robot.set_config(new_goal)
        # a numpy array contains start, waypoints, end
        return result_tmp.solution, {'pos_reset': pos_reset}
    

    def sample_1_val_episode(self, prev_pos, rng_val):
        '''
        In this method, we need to change the robot pos to detect collision.
        Args:
        prev_pos (np): (2,) array of start pos
        '''
        assert (self._get_joint_pose() - prev_pos).sum() < 1e-5, 'in prev pose'
        assert self.robot.no_collision()
        num_planner_fail = 0
        high = self.robot.limits_high # (14,)
        low = self.robot.limits_low # (14,)
        pos_reset = False
        cnt_iter = 0

        while True:
            cnt_iter += 1
            new_goal = rng_val.uniform(low=low, high=high, size=(len(low),))


            if num_planner_fail > 4:
                '''when fail times > 10, when set the starting pos to 0'''
                pos_reset = True
                prev_pos = self.get_robot_free_pose() # reset to 0 + noise
                print(Fore.RED + f'[Process Fail] num_planner_fail: {num_planner_fail}' + Fore.RESET)

            self.robot.set_config(new_goal)
            epi_dist = self.get_epi_dist(prev_pos, new_goal)

            if epi_dist < self.min_episode_distance \
                and num_planner_fail <= 5 \
                and cnt_iter <= self.sample_cnt_limit: # 1000
                continue
            elif self.robot.no_collision():
                # print(f'self.o_env {self.o_env} {id(self.o_env)}')
                # print('new_goal:', np.round(new_goal, 3))
                ## assume a solution exist!
                start_time = time.time()
                # result_tmp = self.planner.plan(self.o_env, prev_pos, new_goal, timeout=('time', self.planner_timeout))
                result_tmp = self.planner_val.plan(self, prev_pos, new_goal, timeout=('time', self.planner_timeout))


                if result_tmp.solution is not None:
                    elapsed_time = time.time() - start_time
                    # print(f'Solution is found in {elapsed_time}s.')
                    break
                else:
                    elapsed_time = time.time() - start_time
                    print(f'Failed to find a solution in {elapsed_time}s, resample a target. # {num_planner_fail}')
                    num_planner_fail += 1


        print(f'cnt {cnt_iter} diff: {np.abs(new_goal - prev_pos).sum()}, len(sol): {len(result_tmp.solution)}')

        self.robot.set_config(new_goal)
        # a numpy array contains start, waypoints, end
        return [ result_tmp.solution[0], result_tmp.solution[-1] ] , elapsed_time, result_tmp.num_collision_check




    def render_1_traj(self, savepath, traj):
        '''
        Return a image with one maze
        '''        
        img = self.renderer.renders(traj, self.maze_size, self.wall_locations, self.wall_hExts)
        if savepath is not None:
            imageio.imsave(savepath, img)
        return img
    
    def render_composite(self, savepath, trajs):
        '''
        return a image with many mazes
        trajs: list of np2d traj
        '''
        ## broadcast array, repeat is deepcopy, not necessary
        n_traj = len(trajs)
        ms_list = self.maze_size[None,].repeat( n_traj, axis=0 )
        cpgrp_list = self.wall_locations[None,].repeat( n_traj, axis=0 )
        hExtgrp_list = self.wall_hExts[None,].repeat( n_traj, axis=0 )
        img_comp = self.renderer.composite(savepath, trajs, ms_list, cpgrp_list, hExtgrp_list)

        return img_comp

    def set_camera_angle(self):
        pass
        
    def render(self):
        """
        Return a snapshot of the current environment
        """        
        self.renderer.plot_maze_bg(self.maze_size, self.wall_locations, self.wall_hExts)
