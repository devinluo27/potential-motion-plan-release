import numpy as np
from pb_diff_envs.robot.abstract_robot import AbstractRobot
from pb_diff_envs.robot.individual_robot import IndividualRobot, trunc
import pybullet as p
import pdb, time

class RobotGroup(AbstractRobot):
    
    '''
    Grouping multiple robots together into a meta-robot
    '''

    def __init__(self, robots, grouping_mask_fn=None, collision_eps=None, **kwargs):
        '''
        grouping mask function aims to assign the collision mask to each robot
        the argument to the function is an instance of the robot group
        it will be called when loading the robots into PyBullet
        '''
        assert np.all([isinstance(robot, IndividualRobot) for robot in robots])
        self.robots: list[IndividualRobot] = robots
        self.num_robots = len(self.robots)
        self.grouping_mask_fn = grouping_mask_fn
        if collision_eps is None:
            collision_eps = min([robot.collision_eps for robot in self.robots])
        limits_low, limits_high = self._get_limits()
        # self.joints = list(range(len(limits_low)))
        super(RobotGroup, self).__init__(limits_low=limits_low, 
                                         limits_high=limits_high, 
                                         collision_eps=collision_eps, **kwargs)

    def _get_limits(self):
        all_limits_low = []
        all_limits_high = []
        for robot in self.robots:
            all_limits_low.extend(robot.limits_low)
            all_limits_high.extend(robot.limits_high)
        return all_limits_low, all_limits_high

    # =====================pybullet module=======================

    def load2pybullet(self, **kwargs):
        '''load robots one by one'''
        item_ids = [robot.load(**kwargs) for robot in self.robots]
        if self.grouping_mask_fn:
            self.grouping_mask_fn(self)
        self.collision_check_count = 0
        return item_ids

    def set_config(self, config, item_ids=None):
        if item_ids is None:
            item_ids = self.item_ids

        ptr = 0
        for item_id, robot in zip(item_ids, self.robots):
            robot.set_config(config[ptr:(ptr+robot.config_dim)])
            ptr += robot.config_dim
        p.performCollisionDetection()
    
    # =====================internal collision check module=======================
    
    def no_collision(self):
        p.performCollisionDetection()
        no_col_list = [] # (2,)
        for item_id in self.item_ids:
            c_pts = p.getContactPoints(item_id)
            
            # sometimes c_pts is None...
            # 1. (c_pts is None) -> no collision
            # 2. is None -> getC again
            if c_pts is None:
                c_pts = p.getContactPoints(item_id)
            no_col = c_pts is None or (len(c_pts) == 0)

            no_col_list.append(no_col)
            if not no_col: # detect collision, no need to continue
                break

        if np.all(no_col_list): # all the list are True
            self.collision_check_count += 1
            return True
        else:
            self.collision_check_count += 1
            return False
        
    def print_joint_states(self):
        # 0: jointPosition
        # 1: jointVelocity
        print(f"---- item_ids:{self.item_ids} joint_states ----")
        pose = self.get_joint_pose()
        # vel = self.get_joint_velocity()
        pose = trunc(pose, 4) # trunc for good format
        # make the array separated by comma
        print(f"Position: {repr(pose)}")
        # print(f"Velocity: {repr(vel)}")
        ptr = 0
        for i_r in range(self.num_robots): # print for each robot
            dim = self.robots[i_r].config_dim # extract config dim
            print(f'robo {i_r}: { repr(pose[ptr:ptr+dim]) }')
            ptr += dim
        print("---------------------------")

    def get_joint_pose(self) -> np.ndarray:
        '''np1d (14,)'''
        states = []
        for i_r in range(self.num_robots):
            pose = self.robots[i_r].get_joint_pose()
            states.append(pose)
        states = np.concatenate(states, axis=0) # e.g. (14,) checked
        return states
    
    def get_end_effector_loc3d(self) -> np.ndarray:
        '''np2d (n_robot, 3)'''
        loc3d_list = []
        for i_r in range(self.num_robots):
            loc3d = self.robots[i_r].get_ee_loc3d()
            loc3d_list.append(loc3d)

        return np.array(loc3d_list)


