from abc import ABC, abstractmethod
import numpy as np
from pb_diff_envs.robot.individual_robot import IndividualRobot
import pybullet as p
import os.path as osp
import pdb
current_dir = osp.dirname(osp.abspath(__file__))
kuka7d_default_urdf = osp.join(current_dir, "../data/robot/kuka_iiwa/model_0.urdf") # abs path

class KukaRobot(IndividualRobot):

    def __init__(self, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1), urdf_file=kuka7d_default_urdf, collision_eps=0.5, **kwargs):
        super(KukaRobot, self).__init__(base_position=base_position, 
                                        base_orientation=base_orientation, 
                                        urdf_file=urdf_file, 
                                        collision_eps=collision_eps, **kwargs)
        self.endEffectorIndex = 6
        self.joints_max_force = self._get_joints_max_force(self.urdf_file) # np array

        
    def _get_joints_and_limits(self, urdf_file):
        pid = p.connect(p.DIRECT)
        item_id = p.loadURDF(urdf_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, physicsClientId=pid)
        num_joints = p.getNumJoints(item_id, physicsClientId=pid)
        limits_low = [p.getJointInfo(item_id, jointId, physicsClientId=pid)[8] for jointId in range(num_joints)]
        limits_high = [p.getJointInfo(item_id, jointId, physicsClientId=pid)[9] for jointId in range(num_joints)]
        p.disconnect(pid)
        return list(range(num_joints)), limits_low, limits_high
    
    def _get_joints_max_force(self, urdf_file):
        '''can be moved to the func above'''
        pid = p.connect(p.DIRECT)
        item_id = p.loadURDF(urdf_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, physicsClientId=pid)
        num_joints = p.getNumJoints(item_id, physicsClientId=pid)
        max_force = [p.getJointInfo(item_id, jointId, physicsClientId=pid)[10] for jointId in range(num_joints)]
        p.disconnect(pid)
        # max_force: [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]
        # print(f'max_force: {max_force}')
        return np.array(max_force)
    
    def load2pybullet(self, **kwargs):
        item_id = p.loadURDF(self.urdf_file, self.base_position, self.base_orientation, useFixedBase=True, **kwargs)
        return item_id
