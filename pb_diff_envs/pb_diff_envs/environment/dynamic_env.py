from abc import ABC, abstractmethod
from pb_diff_envs.environment.abstract_env import AbstractEnv
import pybullet as p
from pb_diff_envs.objects.dynamic_object import DynamicObject


class DynamicEnv(AbstractEnv):
    
    def state_fp(self, state, t):
        return self.robot._state_fp_dynamic(self, state, t)

    def edge_fp(self, state, new_state, t_start, t_end):
        return self.robot._edge_fp_dynamic(self, state, new_state, t_start, t_end)