from pb_diff_envs.environment.static.rand_kuka_env import RandKukaEnv
from pb_diff_envs.environment.static.dualkuka_rand_vgrp import DualKuka_VoxelRandGroupList

class Kuka_VoxelRandGroupList(DualKuka_VoxelRandGroupList):
    '''all function seems to be the same as dual kuka version.'''
    def __init__(self, *args, **kwargs):
        '''
        '''
        print(f'kuka args', args) # is () empty
        print(f'kuka kwargs:', kwargs)
        
        assert kwargs.get('robot_env', None) in [RandKukaEnv,] or args[0] == RandKukaEnv
        super().__init__(*args, **kwargs)
        self.env_type = 'kuka7d'