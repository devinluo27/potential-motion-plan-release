import pybullet as p
from pb_diff_envs.environment.static_env import StaticEnv
from pb_diff_envs.robot.multi_robot.dual_kuka_robot import DualKukaRobot
from pb_diff_envs.robot.grouping import RobotGroup
from pb_diff_envs.environment.abstract_env import load_table
import numpy as np
from pb_diff_envs.environment.offline_env import OfflineEnv
from pb_diff_envs.environment.pybullet_env import PybulletEnv
import time
from pb_diff_envs.planner.bit_star_planner import BITStarPlanner
from pb_diff_envs.objects.obstacles import ObstaclePositionWrapper
import pdb
from operator import itemgetter
from pb_diff_envs.utils.kuka_utils_luo import from_rel2abs_path
from gym.utils import EzPickle
from colorama import Fore

class RandDualKuka14dEnv(StaticEnv, OfflineEnv, PybulletEnv, EzPickle):    
    robot_class = DualKukaRobot

    def __init__(self, objects, robot_config=None, **kwargs):

        self.debug_mode = kwargs.get('debug_mode', False)

        if robot_config is None:
            robot = DualKukaRobot()
        else:
            assert 'base_positions' not in robot_config
            robot = DualKukaRobot(**robot_config, debug_mode=self.debug_mode)
        
        StaticEnv.__init__(self, objects, robot)
        ## add support for picklize the object
        EzPickle.__init__(self, objects=objects, robot_config=robot_config, **kwargs)

        self.img_w = 480
        self.img_h = 360
        fov = 105
        near_plane = 0.1
        far_plane = 100
        self.view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0], distance=2.,
                yaw=-176.64, pitch=-30.31, roll=0, upAxisIndex=2)
        
        self.proj_mat = p.computeProjectionMatrixFOV(fov,  aspect=self.img_w / self.img_h, 
                                                     nearVal=near_plane, farVal=far_plane)

        self.robot: RobotGroup
        assert type(self.robot) == DualKukaRobot
        assert self.robot_class == DualKukaRobot


        PybulletEnv.__init__(self, obs_low=self.robot.limits_low, obs_high=self.robot.limits_high,
                             action_low=np.zeros_like(self.robot.joints_max_force, dtype=np.float32),
                             action_high=self.robot.joints_max_force)
        OfflineEnv.__init__(self, **kwargs)
        

        ## seed not implement yet
        seed_planner = kwargs.get('seed_planner', None)
        assert seed_planner is None
        planner_num_batch = kwargs.get('planner_num_batch', 200)
        self.planner = BITStarPlanner(num_batch=planner_num_batch, stop_when_success=True, seed=seed_planner)
        self.planner_timeout = kwargs.get('planner_timeout', 90) # seconds

        self.planner_val = self.planner

        self.o_env = ObstaclePositionWrapper(self)
        self.wall_locations = self.o_env.get_obstacles()[:, :3] # np (20, 3)
        self.min_episode_distance = kwargs.get('min_episode_distance', 8 * np.pi) # checkparam
        self.sample_cnt_limit = 1e5


        assert (np.abs(self.robot.limits_high - np.array(
            [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05, 
             2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])) < 1e-2).all(), 'check if limit is set.'
        
        self.env_id = 'dualkuka14d'



    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=-176.64,
            cameraPitch=-30.31,
            cameraTargetPosition=[0, 0, 0])
        
    def render(self):
        """
        Return a snapshot of the current environment
        """        
        return p.getCameraImage(width=self.img_w, height=self.img_h, lightDirection=[0, 0, 1], shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL, viewMatrix=self.view_mat, projectionMatrix=self.proj_mat)[2]
    
    def load(self, **kwargs):
        super().load(**kwargs) # load two robots and all obstacles
        ## table path to be set, a larger one
        table = from_rel2abs_path(__file__, '../../data/robot/kuka_iiwa/table_dualkuka14d/table.urdf')
        self.table_id = load_table(table) # should be the last one to load, id is the largest


    def get_target(self):
        return self._target
    
    def get_noisy_pos0(self, low):
        '''low simply provides a shape'''
        start_noisy = np.zeros(shape=(len(low),),) + np.random.randn(len(low),) * 0.01
        return start_noisy

    def sample_1_episode(self, prev_pos):
        '''
        In this method, we need to change the robot pos to detect collision.
        Args:
        prev_pos (np): (7,) array of start pos
        '''
        assert (self._get_joint_pose() - prev_pos).sum() < 1e-5
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
                prev_pos = self.get_noisy_pos0(low) # reset to 0 + noise
                print(Fore.RED + f'[Process Fail] num_planner_fail: {num_planner_fail}' + Fore.RESET)

            self.robot.set_config(new_goal)
            # if np.linalg.norm(new_goal - prev_pos) < 2.0 * np.pi:
            if np.abs(new_goal - prev_pos).sum() < self.min_episode_distance \
                and num_planner_fail <= 5 \
                and cnt_iter <= self.sample_cnt_limit: # 1000
                continue
            elif self.robot.no_collision():
                # print(f'self.o_env {self.o_env} {id(self.o_env)}')
                # print('new_goal:', np.round(new_goal, 3))
                ## assume a solution exist!
                start_time = time.time()
                result_tmp = self.planner.plan(self.o_env, prev_pos, new_goal, timeout=('time', self.planner_timeout))


                if result_tmp.solution is not None:
                    elapsed_time = time.time() - start_time
                    print(f'Solution is found in {elapsed_time}s.')
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
        In contrast to voxel group, we need to add reset to prevent infini sampling.
        Args:
        prev_pos (np): (14,) array of start pos
        Returns: only the start and end, which is feasible
        '''
        assert (self._get_joint_pose() - prev_pos).sum() < 1e-5
        num_planner_fail = 0
        low = self.robot.limits_low
        high = self.robot.limits_high
        pos_reset = False
        cnt_iter = 0

        while True:
            cnt_iter += 1
            new_goal = rng_val.uniform(low=low, high=high, size=(len(low),))

            if num_planner_fail > 4:
                pos_reset = True
                prev_pos = self.get_noisy_pos0(low) # reset to 0 + noise
                print(Fore.RED + f'[Process Fail] num_planner_fail: {num_planner_fail}' + Fore.RESET)
                print(Fore.RED + f'[Process Fail] Reset prev_pos to {prev_pos}' + Fore.RESET)

            self.robot.set_config(new_goal)

            if np.abs(new_goal - prev_pos).sum() < self.min_episode_distance \
                and num_planner_fail <= 5 \
                and cnt_iter <= self.sample_cnt_limit:
                continue
            elif self.robot.no_collision():
                print('new_goal:', np.round(new_goal, 3))
                ## assume a solution exist!
                start_time = time.time()
                result_tmp = self.planner_val.plan(self.o_env, prev_pos, new_goal, timeout=('time', self.planner_timeout))
                if result_tmp.solution is not None:
                    elapsed_time = time.time() - start_time
                    print(f'[Val] Solution is found in {elapsed_time}s.')
                    break
                else:
                    elapsed_time = time.time() - start_time
                    print(f'[Val] Failed to find a solution in {elapsed_time}s, resample a target.')
                    num_planner_fail += 1


        self.robot.set_config(new_goal)
        # a numpy array contains start, end
        return [ result_tmp.solution[0], result_tmp.solution[-1] ], elapsed_time, result_tmp.num_collision_check


    def sample_1_val_episode_no_check(self):
        '''not checked'''
        low = self.robot.limits_low
        high = self.robot.limits_high
        cnt_iter = 0
        while True:
            cnt_iter += 1

            prev_pos = np.random.uniform(low=low, high=high, size=(len(low),))
            self.robot.set_config(prev_pos)
            if not self.robot.no_collision():
                continue

            new_goal = np.random.uniform(low=low, high=high, size=(len(low),))
            self.robot.set_config(new_goal)
            if not self.robot.no_collision():
                continue

            if np.abs(new_goal - prev_pos).sum() > self.min_episode_distance or cnt_iter > 5e5:
                break
        print( f'cnt_iter: {cnt_iter}, reach limits: {cnt_iter > 1e6}' )
        return [ prev_pos, new_goal ], 0, 0


    def set_target(self, target_location=None):    
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()
    
    def _get_obs(self):
        '''
        return numpy array (7,) of the robot's joint
        '''
        return self._get_joint_pose()

    def _get_joint_pose(self) -> np.ndarray:
        '''
        return numpy array (7,) of the robot's joint
        '''
        # print(f'p.getJointStates(self.robot_id, self.robot.joints) {p.getJointStates(self.robot_id, self.robot.joints)}')
        pose = self.robot.get_joint_pose()
        ## many attributes are returned, we only need 0.
        return pose
    
    





    