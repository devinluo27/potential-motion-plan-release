import numpy as np
from .rand_maze2d_env import RandMaze2DEnv
from gym.utils import EzPickle
from colorama import Fore
import time, pdb

class ComposedRM2DEnv(RandMaze2DEnv): # maybe custom an abstraction?
    # robot_class = Point2DRobot

    def __init__(self, wall_locations_c, wall_hExts_c, robot_config, renderer_config={}, **kwargs):
        '''
        env for composing two -> multiple wall lists
        '''
        # (n_, n_w, 3) list
        self.n_models = len(wall_locations_c)
        assert type(wall_locations_c) == list
        assert wall_locations_c[0].ndim == 2 and wall_hExts_c[0].ndim == 2
        
        # nw1+nw2, 2
        wloc_flat = np.concatenate( wall_locations_c, axis=0 )
        hExt_flat = np.concatenate( wall_hExts_c, axis=0 )


        RandMaze2DEnv.__init__(self, wloc_flat, hExt_flat, robot_config, renderer_config, **kwargs)
        self.wloc_list_c = wall_locations_c # list, store wall loc for each model
        self.hExt_list_c = wall_hExts_c # list
        self.env_id = 'composed_static_maze2d'

    