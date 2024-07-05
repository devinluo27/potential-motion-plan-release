import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
import warnings, pdb
from diffuser.datasets.data_api import load_environment
import torch
import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def plot2img(fig, remove_margins=True):
    '''
    convert a plt plot to an img
    '''
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#


from pb_diff_envs.utils.kuka_utils_luo import visualize_kuka_traj_luo
from pb_diff_envs.utils.robogroup_utils_luo import robogroup_visualize_traj_luo
from pb_diff_envs.utils.utils import save_gif as kuka_save_gif
from pb_diff_envs.utils.kuka_utils_luo import SolutionInterp

def transform_obs(observations, maze_idx):
    '''extract observations to np, and maze_idx to index'''
    if type(maze_idx) == np.ndarray:
        maze_idx = maze_idx.astype(np.uint32).item()
    elif type(maze_idx) == torch.Tensor:
        maze_idx = maze_idx.cpu().to(torch.int32).item()
    elif type(maze_idx) in [int, str]:
        pass
    else: 
        raise NotImplementedError

    if torch.is_tensor(observations):
        observations = observations.cpu().numpy()
    return observations, maze_idx

class KukaRenderer:

    def __init__(self, env, is_eval=False, is_sol_kp_mode=False):
        ## when eval, env is a str, checked
        input_env = env
        if type(env) is str: env = load_environment(env, is_eval=is_eval)
        self.env = env
        # assert self.env.is_eval or type(input_env) is str
        assert self.env.is_eval == is_eval
        # pdb.set_trace()
        self.sol_interp = SolutionInterp(density=3) if is_sol_kp_mode else None

    def renders(self, env_single, traj, title=None, fig_dpi=None):
        '''env should not be env list '''
        # pdb.set_trace()
        if traj.shape[-1] == 7:
            gifs, ds, vis_dict = visualize_kuka_traj_luo(env_single, traj)
        elif traj.shape[-1] == 14: # dualkuka14d
            gifs, ds, vis_dict = robogroup_visualize_traj_luo(env_single, traj) # (60,14)

        return gifs, ds, vis_dict

    def composite(self, savepath, paths, ncol=5, only_get_img=False, **kwargs):
        '''
            paths: is the trajectory
            savepath : str
            observations : [ n_paths x horizon x 2 ]
            maze_idx (Required): [n_paths x 1], list or np or torch
        '''
        n_cnt = 0 # may have many paths
        for path, kw in zipkw(paths, **kwargs):
            ## path would be tuple of size 1 ( np(horizon, 7), )
            ## path is tuple of size (1,)
            traj, maze_idx = transform_obs(*path, **kw) # np(240, 7), scalar int
            env_single = self.env.create_single_env(maze_idx)
            if self.sol_interp is not None:
                assert len(traj) < 10, 'make sure to interploate before visualize'
                traj = self.sol_interp(traj)
            # pdb.set_trace()
            gifs, ds, vis_dict = self.renders(env_single, traj)
            new_suffix = f"-sev{vis_dict['se_valid']}_len{len(traj)}ncf{vis_dict['ncoll_frame']}_{n_cnt}.gif"
            savepath_tmp = savepath.replace('.png', new_suffix)
            n_cnt += 1
            kuka_save_gif(gifs, savepath_tmp, duration=ds)







#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)

