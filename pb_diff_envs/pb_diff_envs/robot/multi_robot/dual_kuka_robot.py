from abc import ABC, abstractmethod
import numpy as np
from pb_diff_envs.robot.kuka_robot import KukaRobot, kuka7d_default_urdf
from pb_diff_envs.robot.grouping import RobotGroup
import pybullet as p
import pdb

class DualKukaRobot(RobotGroup):

    def __init__(self, base_positions=((-0.5, 0, 0), (0.5, 0, 0)), 
                       base_orientations=((0, 0, 0, 1), (0, 0, 0, 1)), 
                       urdf_file=kuka7d_default_urdf, 
                       collision_eps=0.5, **kwargs):

        robots = []
        for base_position, base_orientation in zip(base_positions, base_orientations):
            robots.append(KukaRobot(base_position=base_position, base_orientation=base_orientation, urdf_file=urdf_file, collision_eps=collision_eps))

        super(DualKukaRobot, self).__init__(robots=robots, **kwargs)
        self.debug_mode = kwargs.get('debug_mode', False)
        self.set_group_joints_max_force()

    
    def set_group_joints_max_force(self):
        '''
        joint force is also a concat np1d
        '''
        self.joints_max_force = []
        for i in range(len(self.robots)):
            self.joints_max_force.append( self.robots[i].joints_max_force )
        self.joints_max_force: np.ndarray = np.concatenate(self.joints_max_force, axis=0)
        # if self.debug_mode:
            # pdb.set_trace()