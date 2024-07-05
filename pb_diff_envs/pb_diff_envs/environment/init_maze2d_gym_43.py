from gym.envs.registration import register
import numpy as np
from pb_diff_envs.environment.static.rand_maze2d_env import RandMaze2DEnv
from pb_diff_envs.environment.static.comp_rm2d_env import ComposedRM2DEnv
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '..', 'datasets')
eval_ng_300 = 300
eval_np_200 = 200
eval_np_50 = 50
dyn_eval_ng_100 = 100
dyn_eval_np_2000 = 2000

def register_maze2d_concave_43():
    ## Register Maze2D Env with Concave Obstacles

    ## Main
    hr = 0.25
    register(
        id='randSmaze2d-C43-ng3ks25k-ms55nw21-hExt05-v0',
        entry_point='pb_diff_envs.environment.static.maze2d_rand_wgrp_43:Maze2DRandRecGroupList_43',
        max_episode_steps=150,
        kwargs={
            'robot_env': RandMaze2DEnv,
            'num_groups': 3000,
            'samples_per_env': 25000,
            'num_walls': 21,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    # New
                                    min_rep_dist=5e-4,
                                    ball_rad_n=1.8,
                                    min_episode_distance=5.0,
                                    ),

            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 100,
            'dataset_url':f'{root_dir}/randSmaze2d-C43-ng3ks25k-ms55nw21-hExt05-v0.hdf5',
        }
    )




    # -----------------------------------------------
    # ------------------ testOnly -------------------
    # -----------------------------------------------
    ## Concave obstacle: 3 cubes out of 4 smaller blocks
    hr = 0.25
    register(
        id='randSmaze2d-C43-testOnly-v0',
        entry_point='pb_diff_envs.environment.static.maze2d_rand_wgrp_43:Maze2DRandRecGroupList_43',
        max_episode_steps=150,
        kwargs={
            'robot_env': RandMaze2DEnv,
            'num_groups': 12,
            'samples_per_env': 1000,
            'num_walls': 21,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.01?
                                    seed_planner=None,
                                    planner_timeout=30,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,

                                    robot_collision_eps=0.01,
                                    ),
            'eval_num_groups': 6,
            'eval_num_probs': 20,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 3,
            'dataset_url':f'{root_dir}/randSmaze2d-C43-testOnly-v0.hdf5',
        }
    )