from collections import OrderedDict
import os


from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym, pdb

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class PybulletEnv(gym.Env):
    """
    A Dummy Template
    Superclass for all MuJoCo environments.
    """

    def __init__(self, obs_low=-1, obs_high=1, action_low=-1, action_high=1,):

        self._set_action_space(action_low, action_high)

        
        self._set_observation_space(obs_low, obs_high)

        self.seed()

    def _set_action_space(self, low, high):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)
        return self.action_space


    def _set_observation_space(self, obs_low, obs_high):
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=obs_low.shape, dtype=np.float32)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        raise NotImplementedError()
        

    def set_state(self, qpos, qvel):
        raise NotImplementedError()
        
    @property
    def dt(self):
        raise NotImplementedError()
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        raise NotImplementedError()
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        raise NotImplementedError()
        

    def close(self):
        raise NotImplementedError()
        


    def _get_viewer(self, mode):
        raise NotImplementedError()
        
    def get_body_com(self, body_name):
        raise NotImplementedError()


    def state_vector(self):
        raise NotImplementedError()
       