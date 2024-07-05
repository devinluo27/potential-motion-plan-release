import numpy as np
from diffuser.datasets.data_api import load_environment
from .arrays import to_np
import torch, pdb
import matplotlib.pyplot as plt


class RandStaticMazeRenderer:

    def __init__(self, env, observation_dim=None, **kwargs):
        "env should be a maze list"
        self.env_name = env if type(env) == str else env.name
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)


    def renders(self, observations, maze_idx, conditions=None, **kwargs):
        """
        return a numpy image
        if wall_locations_list is given, maze_idx will be ignored.
        kwargs:
            wall_locations_list (a list of np or a np)
        """
        # NOTE get maze layout here!
        ## maze_idx = maze_idx.astype(np.uint32).item() if type(maze_idx) != int else maze_idx
        if type(maze_idx) == np.ndarray:
            maze_idx = maze_idx.astype(np.int32).item()
        elif type(maze_idx) == torch.Tensor:
            maze_idx = maze_idx.cpu().to(torch.int32).item()
        elif type(maze_idx) in [int, str]:
            pass
        else: 
            raise NotImplementedError

    
        observations = to_np(observations)
        raise NotImplementedError()
    

    def composite(self, savepath: str, observations, maze_idx):
        '''
        observations: [n_maze, h, 2]
        maze_idx: [n_maze, 1]
        '''
        maze_idx = to_np(maze_idx)
        maze_idx = np.copy(maze_idx).astype(np.int32) # might be a list
        trajs = to_np( observations )

        ## env is a envist
        env0_single = self.env.create_single_env(maze_idx[0].item())
        if len(np.unique(maze_idx)) == 1: # same maze
            env0_single.render_composite(savepath, trajs)
        else:
            self.env.render_mazelist(savepath, trajs, maze_idx)

            




class RandDynamicMazeRenderer:
    '''
    For rendering a Dynamic Env
    '''
    def __init__(self, env, observation_dim=None, **kwargs):
        "env should be a maze list"
        self.env_name = env if type(env) == str else env.name
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)

    def composite(self, savepath: str, observations: list, maze_idx, wtrajs_list: list):
        '''
        savepath: file path to save the gif
        observations: tensor [n_maze, h, 2] 
        maze_idx: [n_maze, 1] (ignore)
        wtrajs_list: tensor [n_maze, h, nw*2]
        '''
        trajs = observations

        wl_tmp = []
        for wtrajs in wtrajs_list:
            wl_tmp.append( wtrajs.reshape( wtrajs.shape[0], -1, self.env.world_dim) )# h, n_w, dim
        
        self.env.render_mazelist(savepath, trajs, wl_tmp)
        
            

