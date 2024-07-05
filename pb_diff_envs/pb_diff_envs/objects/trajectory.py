from abc import ABC, abstractmethod
import numpy as np


class AbstractTrajectory(ABC):

    @abstractmethod
    def get_spec(self, t):
        raise NotImplementedError  
        
    @abstractmethod
    def set_spec(self, obstacle, t):
        raise NotImplementedError


class WaypointDiscreteTrajectory(AbstractTrajectory):
    '''
    following the waypoints moving in discrete time
    '''
    
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def get_spec(self, t):
        assert isinstance(t, int)
        if t != -1:
            assert 0<=t<=(len(self.waypoints)-1)
        return self.waypoints[t]
        
    def set_spec(self, obstacle, spec):
        obstacle.set_config(spec)
        
        
class WaypointLinearTrajectory(AbstractTrajectory):
    '''
    following the waypoints moving in continuous time
    the motion is linear between adjacent timesteps
    '''
    
    def __init__(self, waypoints, noise_config={}):
        assert type(waypoints) == np.ndarray or type(waypoints) == list
        self.waypoints = waypoints
        self.noisy = len(noise_config) > 0
        self.noise_config = noise_config

    def get_spec(self, t):
        '''impl abstract method
        t (float or int): should be [0, len(waypoints)]
        do linear interp to get traj
        return: 
        np
        '''
        if t == -1 or t >= len(self.waypoints)-1:
            return self.waypoints[-1]
        # if t != -1:
        #     assert 0<=t<=(len(self.waypoints)-1)
        t_prev, t_next = int(np.floor(t)), int(np.ceil(t))
        # print('t', t, t_prev, t_next)
        spec_prev, spec_next = self.waypoints[t_prev], self.waypoints[t_next]
        spec_interp = spec_prev + (spec_next-spec_prev)*(t-t_prev)
        if self.noisy and t > 0:
            spec_interp = spec_interp + np.random.randn(*spec_interp.shape) * self.noise_config['std']

        return spec_interp

    def set_spec(self, obstacle, spec):
        '''impl abstract method
        place the obstace in new/next place one by one
        obstacle ()
        '''
        obstacle.set_config(spec)        
        

        
class WaypointProportionTrajectory(AbstractTrajectory):
    '''
    following the waypoints moving in continuous time
    the motion is linear between adjacent timesteps
    '''
    
    def __init__(self, waypoints, noise_config={}):
        self.waypoints = waypoints
        self.noisy = len(noise_config) > 0
        self.noise_config = noise_config

    def get_spec(self, t):
        '''impl abstract method
        t (float or int): should be [0, len(waypoints)]
        do linear interp to get traj
        return: 
        np
        '''
        if t == -1 or t >= len(self.waypoints)-1:
            return self.waypoints[-1]
        # if t != -1:
        #     assert 0<=t<=(len(self.waypoints)-1)
        t_prev, t_next = int(np.floor(t)), int(np.ceil(t))
        spec_prev, spec_next = self.waypoints[t_prev], self.waypoints[t_next]

        norm_diff = np.linalg.norm(spec_next - spec_prev)
        # print('norm_diff', norm_diff)
        spec_interp = spec_prev + (spec_next-spec_prev)*(t-t_prev)
        if self.noisy and t > 0:
            spec_interp = spec_interp + np.random.randn(*spec_interp.shape) * self.noise_config['std']

        return spec_interp

    def set_spec(self, obstacle, spec):
        '''impl abstract method
        place the obstace in new/next place one by one
        obstacle ()
        '''
        obstacle.set_config(spec)        
        