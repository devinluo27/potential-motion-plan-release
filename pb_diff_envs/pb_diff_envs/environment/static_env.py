from abc import ABC, abstractmethod
from pb_diff_envs.environment.abstract_env import AbstractEnv
import pybullet as p


class StaticEnv(AbstractEnv):

    def state_fp(self, state):
        return self.robot._state_fp(state)
    
    def edge_fp(self, state, new_state):
        return self.robot._edge_fp(state, new_state)