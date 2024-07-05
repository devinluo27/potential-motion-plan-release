import einops, imageio
import numpy as np
import matplotlib.pyplot as plt
from pb_diff_envs.utils.utils import save_gif
from pb_diff_envs.objects.trajectory import WaypointLinearTrajectory
from pb_diff_envs.environment.static.rand_maze2d_renderer import RandMaze2DRenderer
from pb_diff_envs.environment.dynamic_rrgroup import DynamicRectangleWall, DynamicRecWallGroup
from typing import Union
from pb_diff_envs.utils.maze2d_utils import get_is_collision_wtrajs
import torch, pdb

class DynamicRecRenderer:
    def __init__(self, num_walls_c=None) -> None:
        self.static_randerer = RandMaze2DRenderer(num_walls_c, fig_dpi=75)
        self.vis_n_frames = None
        self.min_to_wall_dist = 0.01
        
    

    def render_png_v2(self, savepath, maze_size, r_traj, wtrajs, hExts):
        '''get the rollout first, and then put all of them in one png'''
        images, _, vis_dict = self.get_vis_gif_np_equallen( maze_size, r_traj, wtrajs, hExts )
        images = self.compose_png(images)
        if savepath is not None:
            savepath = savepath.replace('.gif', f"_ncf{vis_dict['ncoll_frame']}.gif")
            imageio.imsave(savepath, images)
        print(f'Saved dyn png to: {savepath}')

        return images # np (500, 2500, 4)

    def render_gif_v2(self, savepath, maze_size, r_traj, wtrajs, hExts):
        '''get the rollout and generate a gif'''
        gifs, ds, vis_dict = self.get_vis_gif_np_equallen( maze_size, r_traj, wtrajs, hExts )
        if savepath is not None:
            savepath = savepath.replace('.gif', f"_ncf{vis_dict['ncoll_frame']}.gif")
            save_gif(gifs, savepath, duration=ds)
        
        return gifs # a list of imgs
        

    def get_vis_gif_np_equallen(self, maze_size, 
                    r_traj: np.ndarray,
                    wtrajs: np.ndarray,
                    hExts: np.ndarray,
                    ):
        ''' returns a sequence of images along time
        dyn_wall_grp: a group of dyn walls
        robot traj:
        wtrajs: accept two shape: 1. (h, nw, dim) 2. (nw, h, dim)
            but renderer needs: (nw, h, dim)
        '''
        # get the max len of all dyn walls (must align same len)
        # max_len_obj_traj = dyn_wall_grp.get_max_len_wtraj()
        if wtrajs.shape[0] == r_traj.shape[0] and wtrajs.shape[1] == hExts.shape[0]:
            if torch.is_tensor(wtrajs): 
                wtrajs = wtrajs.cpu().numpy()
            wtrajs = wtrajs.transpose(1, 0, 2)
        assert wtrajs.ndim == 3
        assert wtrajs.shape[1] == r_traj.shape[0], 'same len'
        n_frames = r_traj.shape[0]

        is_cols = get_is_collision_wtrajs(r_traj, wtrajs, hExts, min_to_wall_dist=self.min_to_wall_dist)
        ncf = is_cols.sum()
        has_col = ncf > 0
        
        print(f'[dyn render] wtrajs {wtrajs.shape}, r_traj: {r_traj.shape}, ')
        gifs = []
        for timestep in range(n_frames):
            pose_robot = r_traj[timestep]
            ## iteratively set wall pose
            # for obj in dyn_wall_grp.dyn_recWall_list:
                # obj.set_config_at_time(timestep)
            # p.performCollisionDetection()
            centers = wtrajs[:, timestep, :] # (n_w, 2)
            # img = self.static_randerer.get_img_maze_bg(maze_size, centers, hExts)
            img = self.static_randerer.get_img_maze_bg_with_wtrajs(maze_size, centers, hExts, wtrajs=wtrajs, robo_traj=r_traj, pose_robot=pose_robot, is_col=is_cols[timestep], has_col=has_col,)
            # print('img', img.shape)
            gifs.append(img)




        ds = [150] * len(gifs)
        ds[0] = 1000; ds[-1] = 3000
        vis_dict = dict(ncoll_frame=ncf,) # se_valid=se_valid)
        return gifs, ds, vis_dict
    

    def compose_png(self, images):
        '''reshape a list of images and generate one large image'''
        n_col = min( len(images), 5)
        residual = len(images) % n_col
        ## padding
        if residual > 0:
            images.extend( [ np.ones_like(images[0]), ] * (5 - residual)  )
        images = np.stack(images, axis=0)
        nrow = len(images) // n_col
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=n_col)
        return images