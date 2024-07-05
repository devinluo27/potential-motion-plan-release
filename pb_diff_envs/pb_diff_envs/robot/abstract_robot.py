from abc import ABC, abstractmethod
import numpy as np
from pb_diff_envs.environment.dynamic_rrgroup import AbsDynamicWall
from pb_diff_envs.objects.dynamic_object import MovableObject, DynamicObject
from pb_diff_envs.objects.trajectory import AbstractTrajectory
import pybullet as p
import pdb

class AbstractRobot(MovableObject, ABC):

    # An abstract robot    
    def __init__(self, limits_low, limits_high, collision_eps, **kwargs):
        self.limits_low = np.array(limits_low, dtype=np.float32).reshape(-1)
        self.limits_high = np.array(limits_high, dtype=np.float32).reshape(-1)
        self.collision_eps = collision_eps
        self.config_dim = len(self.limits_low)
        super(AbstractRobot, self).__init__(**kwargs)

    # =====================pybullet module=======================
    
    @abstractmethod
    def set_config(self, config, item_id=None):
        '''
        set a configuration
        '''        
        raise NotImplementedError        
        
    # =====================sampling module=======================        
        
    def uniform_sample(self, n=1, rng:np.random.BitGenerator=None):
        '''
        uniform sample in the range of robot limit
        '''
        if rng is None:
            sample = np.random.uniform(self.limits_low.reshape(1, -1), self.limits_high.reshape(1, -1), (n, self.config_dim))
        else:
            sample = rng.uniform( self.limits_low.reshape(1, -1), self.limits_high.reshape(1, -1), (n, self.config_dim) )

        if n==1:
            return sample.reshape(-1)
        else:
            return sample

    def sample_free_config(self):
        while True:
            state = self.uniform_sample()
            if self._state_fp(state):
                return state        
        
    def sample_n_points(self, n, need_negative=False):
        positive = []
        negative = []
        for i in range(n):
            while True:
                state = self.uniform_sample()
                if self._state_fp(state):
                    positive.append(state)
                    break
                else:
                    negative.append(state)
        if not need_negative:
            return positive
        else:
            return positive, negative

    def sample_n_free_points(self, n):
        positive = []
        cnt = 0
        while cnt < n:
            state = self.uniform_sample()
            if self._state_fp(state):
                positive.append(state)
                cnt += 1

        return positive


    def sample_random_init_goal(self):
        while True:
            init, goal = self.sample_free_config(), self.sample_free_config()
            if np.sum(np.abs(init - goal)) != 0:
                break
        return init, goal
    
    def in_goal_region(self, state, goal):
        return np.linalg.norm(state-goal) < self.collision_eps
    
    # =====================internal collision check module=======================
    
    @abstractmethod
    def no_collision(self):
        '''
        Perform the collision detection
        '''
        raise NotImplementedError      
    
    def _valid_state(self, state):
        '''check if reach the limit of the robot'''
        return (state >= np.array(self.limits_low)).all() and \
               (state <= np.array(self.limits_high)).all()      
    
    def _state_fp(self, state):
        ''' three steps:
        1. check if given state is valid
        2. set to given state
        3. check collision
        '''
        # return false if state out of limit
        if not self._valid_state(state):
            print('invalid robot state')
            return False

        ## TODO not known
        self.set_config(state)
        ## TODO not known
        return self.no_collision()
    
    def _iterative_check_segment(self, left, right):
        if np.sum(np.abs(left - left)) > 0.1:
            mid = (left + right) / 2.0
            if not self._state_fp(mid):
                return False
            return self._iterative_check_segment(left, mid) and self._iterative_check_segment(mid, right)

        return True 
    
    def _edge_fp(self, state, new_state):
        '''
        check if any collision happens in the trajectory
        this method is theoretial computation, not real pybullet simulation.
        '''
        assert state.size == new_state.size # in tensor, size is a method, so different

        # 1. the new state and the old state should be valid
        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        # 2. check if collision in start/end points
        if not self._state_fp(state) or not self._state_fp(new_state):
            return False

        disp = new_state - state

        d = self.distance(state, new_state)
        # kuka robot eps 0.5 by default
        K = int(d / self.collision_eps)
        # print(state, new_state)
        # print(f'K {K}, d {d}, self.collision_eps {self.collision_eps}')
        # for k in range(0, K):
        for k in range(1, K): # Jan23, 2024
            c = state + k * 1. / K * disp
            if not self._state_fp(c):
                return False
        return True
    
    def distance(self, from_state, to_state):
        '''
        Distance metric, L2 Norm
        '''
        to_state = np.maximum(to_state, np.array(self.limits_low))
        to_state = np.minimum(to_state, np.array(self.limits_high))
        diff = np.abs(to_state - from_state)

        return np.sqrt(np.sum(diff ** 2, axis=-1))   
    
    # =====================internal collision check module for dynamic environment=======================    
    
    def _state_fp_dynamic(self, env, state, t):
        # pdb.set_trace()
        # loop through each objs: objs can be static or dynamic
        for each_object in env.objects:
            if isinstance(each_object, DynamicObject) or isinstance(each_object, AbsDynamicWall):
                ## maybe setting the location at t?
                each_object.set_config_at_time(t)
            else:
                raise NotImplementedError()
        return self._state_fp(state)
    
    def _edge_fp_dynamic(self, env, state, new_state, t_start, t_end):
        '''
        same structure as _edge_fp, but with _state_fp_dynamic instead of _state_fp
        Return: True if no collision
        '''
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._state_fp_dynamic(env, state, t_start) or not self._state_fp_dynamic(env, new_state, t_end):
            return False

        disp = new_state - state
        t_disp = t_end - t_start

        d = self.distance(state, new_state)
        K = int(d / self.collision_eps)
        # print(f'd {d}, K {K} disp: {disp}; t_start: {t_start}, t_disp: {t_disp}')
        # import time
        # time.sleep(0.1)
        # assume same speed move
        # K = K + 1 if K != 0 else K
        # for k in range(0, K+1):
        for k in range(0, K):
            c = state + k * 1. / K * disp
            t_c = t_start + k * 1. / K * t_disp
            if not self._state_fp_dynamic(env, c, t_c):
                return False
        return True             


class DynamicRobotFactory:
    
    @staticmethod
    def create_dynamic_robot_class(Robot):
        '''
        return a class (a wrapper)
        the class makes robot a dynamic object by inheritance
        so a trajectoy to move the robot is required to init
        '''
        class DynamicRobot(Robot, DynamicObject):
            def __init__(self, trajectory: AbstractTrajectory, **kwargs):
                super(DynamicRobot, self).__init__(item=self, trajectory=trajectory, **kwargs)
        return DynamicRobot