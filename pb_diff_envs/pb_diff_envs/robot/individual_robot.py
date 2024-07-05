from abc import ABC, abstractmethod
import numpy as np
from pb_diff_envs.robot.abstract_robot import AbstractRobot
import pybullet as p
import time, torch, pdb
from operator import itemgetter

class IndividualRobot(AbstractRobot, ABC):
    # An individual robot
    def __init__(self, base_position, base_orientation, urdf_file, **kwargs):
        # for loading the pybullet
        self.base_position = base_position
        self.base_orientation = base_orientation
        
        self.urdf_file = urdf_file

        joints, limits_low, limits_high = self._get_joints_and_limits(self.urdf_file)
        self.joints = joints # a list e.g., [0,1,...,9]
        
        kwargs['base_position'] = base_position
        kwargs['base_orientation'] = base_orientation
        super(IndividualRobot, self).__init__(limits_low=limits_low, 
                                              limits_high=limits_high, **kwargs)

    @abstractmethod
    def _get_joints_and_limits(self, urdf_file):
        raise NotImplementedError

    # @abstractmethod
    # def _get_joints_max_force(self, urdf_file):
    #     raise NotImplementedError
    
    # =====================pybullet module=======================        
        
    def load(self, **kwargs):
        item_id = self.load2pybullet(**kwargs)
        self.collision_check_count = 0
        self.item_id = item_id
        return item_id  
    
    @abstractmethod
    def load2pybullet(self, **kwargs):
        '''
        load into PyBullet and return the id of robot
        '''        
        raise NotImplementedError
        
    def set_config(self, config, item_id=None):
        '''
        set a configuration
        set every given joint state  
        will be used in WaypointLinearTrajectory, to set config at time t
        '''            
        if item_id is None:
            item_id = self.item_id
        ## ----- ori forloop impl -----
        # for i, c in zip(self.joints, config):
        #     p.resetJointState(item_id, i, c)


        ## ----- luo impl (faster) -----
        # print('config', type(config), config) # list or np
        if type(config[0]) == float:
            config = [[c,] for c in config]
        elif type(config) == np.ndarray:
            config = config.reshape(-1, 1)
        elif torch.is_tensor(config):
            config = config.reshape(-1, 1)
        else: 
            raise NotImplementedError()
        p.resetJointStatesMultiDof(item_id, np.arange(len(config)), config)

        p.performCollisionDetection()  
        # time.sleep(0.2)
        
    # =====================internal collision check module=======================

    def no_collision(self):
        '''
        Perform the collision detection
        seems like it only detect self collision? no no
        it only wants to get collision related to the robot
        '''
        p.performCollisionDetection()
        c_pts = p.getContactPoints(self.item_id)
        if c_pts is None:
            c_pts = p.getContactPoints(self.item_id)
        ##
        if (c_pts is None) or (len(c_pts) == 0):
            self.collision_check_count += 1
            return True
        else:
            # print('contact points:', p.getContactPoints(self.item_id))
            self.collision_check_count += 1
            return False

    def get_workspace_observation(self):
        '''
        Get the workspace observation
        '''
        raise NotImplementedError
    
    def print_joint_states(self):
        # 0: jointPosition
        # 1: jointVelocity
        ## Not elegant impl
        # states = np.array( p.getJointStates(self.item_id, self.joints) )
        # states = states[:, :2].astype(np.float32)
        # print(f'states len {len(states)}', type(states)) # tuple

        print(f"---- {self.item_id} joint_states ----")
        pose = self.get_joint_pose()
        vel = self.get_joint_velocity()
        # pose = trunc(pose, 4) # trunc for good format
        # make the array separated by comma
        print(f"Position: {repr(pose)}")
        print(f"Velocity: {repr(vel)}")
        # print(f"jointReactionForces: {s[2]}")
        print("---------------------------")

    def get_joint_pose(self) -> np.ndarray:
        states = p.getJointStates(self.item_id, self.joints)
        pose = np.array( list(map(itemgetter(0), states)), dtype=np.float32 )
        return pose
    
    def get_joint_velocity(self) -> np.ndarray:
        states = p.getJointStates(self.item_id, self.joints)
        velocity = np.array( list(map(itemgetter(1), states)), dtype=np.float32 )
        return velocity

    def get_ee_loc3d(self):
        ''' get the current end effector 3d location
        ee - end effector
        pose: list or np1d
        '''
        # assert len(pose) == self.config_dim
        # ori_pose = self.get_joint_pose()
        # self.set_config(pose)
        loc3d = p.getLinkState(self.item_id, self.endEffectorIndex)[0]
        # self.set_config(ori_pose) ## set back to original pose
        return loc3d


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)