import numpy as np
import einops
import imageio
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pb_diff_envs.utils.diffuser_utils import plot2img
from pb_diff_envs.utils.maze2d_utils import get_is_collision_static, get_is_connected, compute_dist_sum

class RandMaze2DRenderer:
    def __init__(self, num_walls_c=None, fig_dpi=120, up_right=None, middle=None, config={}):
        self._remove_margins = False
        self.fig_dpi = fig_dpi

        self.bg_color = np.array([252, 242, 212, 170]) / 255.0
        self.dw_color = ['#3d405b', '#99627A', '#F2C5E0', '#603A22', '#F8BD7F',
                         '#16E207', '#2E532B', '#6C33D5']
        self.num_walls_c = num_walls_c # for render different color when composing
        self.comp_color = ['#1f77b4D8', '#FF8400CC', '#54F1B8CC'] # CC: alpha=0.8

        self.up_right = up_right # draw the true dist?
        self.short_text = (middle != None) # use is_success or success
        self.middle = middle # redner something in the upper middle of the maze
        self.config: dict = config



    def plot_maze_bg(self, maze_size, center_pos_list, hExt_list): # -> np.ndarray:
        ''' evereything assume unnormed
        given maze size, rec center, rec hExt to render '''
        assert type(maze_size) in [np.ndarray, int]
        plt.clf()
        fig = plt.gcf()
        # canvas_size = maze_size[0] if type(maze_size) == np.ndarray else maze_size # int
        if type(maze_size) == np.ndarray: 
            x_size = maze_size[0]
            y_size = maze_size[1]
        else:
            x_size = y_size = maze_size

        # fig.set_size_inches(x_size, y_size) # 5,5 by default
        fig.set_size_inches(5, 5)
        # if fig_dpi:
        fig.set_dpi(self.fig_dpi)
        plt.xlim(0, x_size)
        plt.ylim(0, y_size)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_facecolor(self.bg_color)
        # plt.tight_layout()
        subp_gap = 0.070
        plt.subplots_adjust(left=subp_gap, right=1-subp_gap, bottom=subp_gap, top=1-subp_gap)


        n_walls = len(center_pos_list)
        for n_w in range(n_walls):
            # bottom left location
            x = center_pos_list[n_w][0] - hExt_list[n_w][0]
            y = center_pos_list[n_w][1] - hExt_list[n_w][1]
            x_len = hExt_list[n_w][0] * 2
            y_len = hExt_list[n_w][1] * 2
            # print((x, y), x_len, y_len)
            patch = plt.Rectangle((x, y), x_len, y_len, fill=True, color=self.comp_color[0])
            if self.num_walls_c is not None:
                c_id = np.searchsorted( np.cumsum(self.num_walls_c), n_w, side='right').item()
                patch.set_color(self.comp_color[c_id])
            plt.gca().add_patch(patch)

        # Customize x and y tick positions and labels
        plt.xticks(range(x_size+1)) # Denser ticks; no ticks []
        plt.yticks(range(y_size+1)) 
        if True:
            # clear tick
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(axis='both', length=0)
            

        # Set the linewidth for all axes at once
        linewidth = 2.0 # 2.0
        for spine in plt.gca().spines.values():
            spine.set_linewidth(linewidth)
            spine.set_color( (0.11, 0.11, 0.11, 0.8 ) )
        
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, linestyle=(0, (5, 5)), linewidth=1.3) # '--'

        # plt.show()
        # img = plot2img(fig, remove_margins=False)
        return fig
    

    ## --------------- For dynamic Rendering -----------------

    def get_img_maze_bg(self, *args):
        '''returns a np (h,w,4)'''
        fig = self.plot_maze_bg(*args)
        return plot2img(fig, remove_margins=False)
    
    def get_img_maze_bg_with_wtrajs(self, *args, wtrajs, robo_traj=None, 
                                    pose_robot=None, is_col=None, has_col=None):
        fig = self.plot_maze_bg(*args)
        assert wtrajs.shape[0] == len(args[1])
        fig = self._add_wtrajs_to_fig(fig, wtrajs)

        if robo_traj is not None:
            fig = self._add_traj_to_fig(robo_traj, fig)
        if pose_robot is not None:
            fig = self._add_1_point_to_fig(fig, pose_robot)

        if is_col is not None:
            fig = self._add_collision_marker(fig, is_col, has_col)

        img = plot2img(fig, remove_margins=False)
        plt.close(fig)
        return img

    
    def _add_wtrajs_to_fig(self, fig: Figure, trajs_wall):
        """
        trajs_wall: n_w, n_pts, 2
        """
        # print(type(fig.gca()))
        assert trajs_wall.ndim == 3 and trajs_wall.shape[-1] == 2
        for i_wt in range(len(trajs_wall)):
            ## no need to plot wtraj for the second models
            if self.num_walls_c is not None and i_wt < self.num_walls_c[0]:
                fig.gca().plot(trajs_wall[i_wt, :, 0], trajs_wall[i_wt, :, 1], 
                            's--', linewidth=2.0, markersize=4,
                            color=self.dw_color[i_wt], alpha=0.5, zorder=5)
                            #    marker='x', linestyle='--', color)

        return fig
    
    def _add_1_point_to_fig(self, fig: Figure, pt):
        '''dynamic, 
        robot position of the current time'''
        assert pt.shape ==(2,)
        fig.gca().scatter(x=pt[0], y=pt[1], marker=(5,2), 
                          zorder=20, s=250, color='#FB4570')
        return fig
    
    def _add_collision_marker(self, fig, is_col, has_col):
        '''add two small block on top row'''
        # is_col = False
        red_c = (0.85, 0, 0, 0.80)
        green_c = (0, 0.85, 0, 0.80)
        if is_col is None:
            pass
        elif is_col:
            fig.text(0.69, 0.982, ' C ', ha='right', va='top', fontsize=12, # 0.7, 0.948, 11
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor=red_c))
        else:
            fig.text(0.69, 0.982, '   ', ha='right', va='top', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor=green_c))
        
        if self.middle == None:
            fig.text(0.08, 0.995, 'is success:', ha='left', va='top', fontsize=23,) # 0.12,0.95
            mark_loc = (0.45, 0.982)
        else:
            ## use a short text description if extra text in the middle
            fig.text(0.07, 0.995, 'success:', ha='left', va='top', fontsize=23,) # 0.12,0.95
            mark_loc = (0.355, 0.982)

        fc = red_c if has_col else green_c
        fig.text(mark_loc[0], mark_loc[1], '   ', ha='left', va='top', fontsize=12, # 0.38, 0.948
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor=fc))
        return fig


    
    ## --------------- Static Rendering -----------------

    def _add_traj_to_fig(self, traj: np.ndarray, fig):
        '''
        plot one traj
        '''
        assert traj.ndim == 2 and traj.shape[1] == 2, f'{traj.shape}'
        path_length = len(traj)
        # NOTE earlier observation 0 -> later 1: blue is start, red is the end
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        ## ** x, y order different with maze2d! **
        if self.config.get('search_tree'):
            st_states = self.config.get('search_tree').states
            colors = plt.cm.jet(np.linspace(0,1,len(st_states)))
            plt.scatter(st_states[:,0], st_states[:,1], c=colors, zorder=20, s=12)
        
        else:
            if self.config.get('no_draw_traj_line', False):
                '''ignore the black line that connects adjacent waypoint'''
                pass
            else:
                plt.plot(traj[:,0], traj[:,1], c='black', zorder=10)
            plt.scatter(traj[:,0], traj[:,1], c=colors, zorder=20, s=12)

        plt.scatter(traj[0,0], traj[0,1], color='#009900', marker='*', s=600, zorder=30, alpha=0.5)
        plt.scatter(traj[-1,0], traj[-1,1], color='#FF3399', marker='*', s=600, zorder=30, alpha=0.5)
        # img = plot2img(fig, remove_margins=False)
        
        if self.up_right == 'dist':
            '''show distance'''
            dist = compute_dist_sum(traj)
            dist = f'{dist:.1f}' if dist <= 9 else f'>9'
            fig.text(0.935, 0.995, f'dist:{dist}', ha='right', va='top', fontsize=23,) # plot len 0.9,0.95
        elif self.up_right == 'empty':
            ## do not even print the len: 48
            pass
        else:
            '''default show # of waypoints'''
            fig.text(0.92, 0.995, f'len:{len(traj)}', ha='right', va='top', fontsize=23,) # plot len 0.9,0.95

        return fig
    
    def _add_middle_text(self, fig, infos):
        if self.middle is None:
            return fig
        fig.text(0.65, 0.990, f"{self.middle}:{infos[self.middle]:.2f}", ha='right', va='top', fontsize=22,)
        return fig
        

    def renders(self, traj, maze_size, center_pos_list, hExt_list, infos=None):
        '''static, render one traj'''
        fig = self.plot_maze_bg(maze_size, center_pos_list, hExt_list)
        fig = self._add_traj_to_fig(traj, fig)
        
        if self.config.get('no_draw_col_info', False):
            pass
        else:
            is_col = get_is_collision_static(traj, center_pos_list, hExt_list)
            not_suc = (is_col.sum()>0) or ( not get_is_connected( traj ) )
            fig = self._add_collision_marker(fig, None, has_col=not_suc )

        fig = self._add_middle_text(fig, infos)

        img = plot2img(fig, remove_margins=False)
        plt.close(fig)
        plt.close('all')
        return img
    


    def composite(self, savepath, paths, 
                  ms_list, cpgrp_list, hExtgrp_list,
                  only_get_img=False, **kwargs):
        '''
            savepath : str
            paths : list of n_paths [ horizon x 2 ], we should generalize to different path len
            ms_list : a list of maze_size

            maze_idx: [n_paths x 1], list or np or torch
        '''
        ncol = min( len(paths), 5) # 4
        assert len(paths) % ncol == 0, f'Number of paths ({len(paths)}) must be divisible by number of columns'
        images = []
        assert type(paths) in [list, np.ndarray]
        assert paths[0].ndim == 2 and ms_list.ndim == 2
        assert cpgrp_list.ndim == 3 and hExtgrp_list.ndim == 3

        self.check_args(paths, ms_list, cpgrp_list, hExtgrp_list,)
        ## smart code: iterate through paths using path
        # for path, kw in zipkw(paths, **kwargs):
        for i_p in range(len(paths)):
            img = self.renders(paths[i_p], ms_list[i_p], cpgrp_list[i_p], hExtgrp_list[i_p])
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        if savepath is not None:
            imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')
        return images # np (500, 2500, 4)


    def check_args(self, *args, last_dim=None):
        for arg in args:
            assert len(arg) == len(args[0])


    