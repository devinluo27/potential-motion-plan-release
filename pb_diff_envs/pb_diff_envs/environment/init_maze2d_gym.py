from gym.envs.registration import register
import numpy as np
from pb_diff_envs.environment.static.rand_maze2d_env import RandMaze2DEnv
from pb_diff_envs.environment.dynamic.dyn_rm2d_env import DynamicRandMaze2DEnv
from pb_diff_envs.environment.static.comp_rm2d_env import ComposedRM2DEnv
from pb_diff_envs.environment.dynamic.comp_stdyn_rm2d_env import ComposedStDynRandMaze2DEnv
import os, pdb


cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '..', 'datasets')

eval_ng_300 = 300
eval_np_200 = 200
dyn_eval_ng_100 = 100
dyn_eval_np_2000 = 2000
multi_w_train_ng = 300
multi_w_train_spe = 25000

def register_maze2d():
    ## Register Maze2D Env with Square Obstacles
    

    ## --------------------------------------------------
    ## -----------------   static   ---------------------
    ## --------------------------------------------------

    hr = 0.50
    register(
        id='randSmaze2d-ng3ks25k-ms55nw6-hExt05-v0',
        entry_point='pb_diff_envs.environment.static.maze2d_rand_wgrp:Maze2DRandRecGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': RandMaze2DEnv,
            'num_groups': 3000,
            'samples_per_env': 25000,
            'num_walls': 6,
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
                                    min_rep_dist=1e-3,
                                    ball_rad_n=1.6,
                                    ),
                                    # min_episode_distance # default
                                    # robot_collision_eps # default
                                    # epi_dist_type
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 100,
            'dataset_url':f'{root_dir}/randSmaze2d-ng3ks25k-ms55nw6-hExt05-v0.hdf5',
        }
    )


    ## larger wall
    hr = 0.7
    register(
        id='randSmaze2d-ng3ks35k-ms55nw3-hExt07-v0',
        entry_point='pb_diff_envs.environment.static.maze2d_rand_wgrp:Maze2DRandRecGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': RandMaze2DEnv,
            'num_groups': 3000,
            'samples_per_env': 35000,
            'num_walls': 3,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 150,
            'dataset_url':f'{root_dir}/randSmaze2d-ng3ks35k-ms55nw3-hExt07-v0.hdf5',
        }
    )


    



    ## --------------------------------------------------
    ## ------------------ Dynamic -----------------------
    ## --------------------------------------------------


    hr = 1.0
    register(
        id='DynrandSmaze2d-ng1ks10k-ms55nw1-hExt10-v0',
        entry_point='pb_diff_envs.environment.dynamic.dyn_rm2d_wgrp:DynRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': DynamicRandMaze2DEnv,
            'num_groups': 1000,
            'samples_per_env': 10000,
            'num_walls': 1,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': False,
            'num_sample_pts': 2,
            'speed': 0.12,
            'mazelist_config': dict(interp_density=None,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.01?
                                    seed_planner=None,
                                    planner_timeout=300,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=1200, # ?
                                    robot_collision_eps=0.04, # ?
                                    k_nb=50,),
            'eval_num_groups': dyn_eval_ng_100,
            'eval_num_probs': dyn_eval_np_2000,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 1000,
            'dataset_url':f'{root_dir}/DynrandSmaze2d-ng1ks10k-ms55nw1-hExt10-v0.hdf5',
        }
    )

    




    
    ## -----------------------------------------------------
    ## ------------------- Composed ------------------------
    ## -----------------------------------------------------

    ## Diff Env static 1 + static 2: 6 + 1, static, composed
    nw1 = 6; nw2 = 1
    hr1 = 0.5; hr2 = 0.7; ct = 'direct'
    register(
        id='ComprandSmaze2d-ms55nw6nw1-hExt0507-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=False,
                                    ),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw1-hExt0507-v0.hdf5',
        }
    )

    ## Diff env 6 + 2, static, composed
    nw1 = 6; nw2 = 2
    hr1 = 0.5; hr2 = 0.7; ct = 'direct'
    register(
        id='ComprandSmaze2d-ms55nw6nw2-hExt0507-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=False,
                                    ),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw2-hExt0507-v0.hdf5',
        }
    )

    ## Diff env 6 + 3, static, composed
    nw1 = 6; nw2 = 3
    hr1 = 0.5; hr2 = 0.7; ct = 'direct'
    register(
        id='ComprandSmaze2d-ms55nw6nw3-hExt0507-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=False,
                                    ),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw3-hExt0507-v0.hdf5',
        }
    )


    ## --------------------------------------------------------
    ## --------------- Over More Obstacles --------------------  
    ## --------------------------------------------------------

    # composed 6 + 1
    nw1 = 6; nw2 = 1
    hr1 = 0.5; hr2 = 0.5; ct = 'share'
    register(
        id='ComprandSmaze2d-ms55nw6nw1-hExt0505-gsdiff-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=True,),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw1-hExt0505-gsdiff-v0.hdf5',
        }
    )

    # composed 6 + 2, use different seed across rand grp
    nw1 = 6; nw2 = 2
    hr1 = 0.5; hr2 = 0.5; ct = 'share'
    register(
        id='ComprandSmaze2d-ms55nw6nw2-hExt0505-gsdiff-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=True,),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw2-hExt0505-gsdiff-v0.hdf5',
        }
    )

    # composed 6 + 3, use different seed across rand grp
    nw1 = 6; nw2 = 3
    hr1 = 0.5; hr2 = 0.5; ct = 'share'
    register(
        id='ComprandSmaze2d-ms55nw6nw3-hExt0505-gsdiff-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=True,),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw3-hExt0505-gsdiff-v0.hdf5',
        }
    )

    # composed 6 + 4
    nw1 = 6; nw2 = 4
    hr1 = 0.5; hr2 = 0.5; ct = 'share'
    register(
        id='ComprandSmaze2d-ms55nw6nw4-hExt0505-gsdiff-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=True,),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw4-hExt0505-gsdiff-v0.hdf5',
        }
    )

    # composed 6 + 5
    nw1 = 6; nw2 = 5
    hr1 = 0.5; hr2 = 0.5; ct = 'share'
    register(
        id='ComprandSmaze2d-ms55nw6nw5-hExt0505-gsdiff-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=True,),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw5-hExt0505-gsdiff-v0.hdf5',
        }
    )

    # composed 6 + 6
    nw1 = 6; nw2 = 6
    hr1 = 0.5; hr2 = 0.5; ct = 'direct'
    register(
        id='ComprandSmaze2d-ms55nw6nw6-hExt0505-gsdiff-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': multi_w_train_ng, # not used
            'n_comp': 2,
            'samples_per_env': multi_w_train_spe, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=True,),
            'eval_num_groups': eval_ng_300,
            'eval_num_probs': eval_np_200,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 50, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-ms55nw6nw6-hExt0505-gsdiff-v0.hdf5',
        }
    )

    # --------------------------------
    # --------------------------------





    ## dynamic+staic, composed
    nw1 = 1; nw2 = 3
    hr1 = 1.0; hr2 = 0.7; ct = 'direct'
    register(
        id='CompDynrandSmaze2d-ms55nw1nw3-hExt1007-v0',
        entry_point='objects.comp.comp_stdyn_m2d_wgrp:ComposedStDynRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedStDynRandMaze2DEnv,
            'num_groups': 100,
            'n_comp': 2,
            'samples_per_env': 2000,
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': False,
            'num_sample_pts': 2,
            'speed': 0.12,
            'wall_is_dyn': [True, False, False, False],
            'mazelist_config': dict(interp_density=None,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=300,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=800, # 1000?
                                    robot_collision_eps=0.04, # ?
                                    k_nb=50,
                                    grp_seed_diff=True,
                                    ),
            'eval_num_groups': dyn_eval_ng_100,
            'eval_num_probs': dyn_eval_np_2000,
            'seed': 333 + 1,
            'eval_seed_offset': 27,
            'gen_num_parts': 100,
            'dataset_url':f'{root_dir}/CompDynrandSmaze2d-ms55nw1nw3-hExt1007-v0.hdf5',
        }
    )

    ## dynamic+staic, composed
    nw1 = 1; nw2 = 6
    hr1 = 1.0; hr2 = 0.5; ct = 'direct'
    register(
        id='CompDynrandSmaze2d-ms55nw1nw6-hExt1005-v0',
        entry_point='objects.comp.comp_stdyn_m2d_wgrp:ComposedStDynRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedStDynRandMaze2DEnv,
            'num_groups': 100,
            'n_comp': 2,
            'samples_per_env': 2000,
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': False,
            'num_sample_pts': 2,
            'speed': 0.12,
            'wall_is_dyn': [True] + [False,] * nw2,
            'mazelist_config': dict(interp_density=None,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=300,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=800, # 1000?
                                    robot_collision_eps=0.04, # ?
                                    k_nb=50,
                                    grp_seed_diff=True,
                                    ),
            'eval_num_groups': dyn_eval_ng_100,
            'eval_num_probs': dyn_eval_np_2000,
            'seed': 333 + 1,
            'eval_seed_offset': 27,
            'gen_num_parts': 100,
            'dataset_url':f'{root_dir}CompDynrandSmaze2d-ms55nw1nw6-hExt1005-v0.hdf5',
        }
    )
    











    # ----------------------------------------------
    # ------------------ luotest -------------------

    hr = 0.50
    register(
        id='randSmaze2d-testOnly-v0',
        entry_point='pb_diff_envs.environment.static.maze2d_rand_wgrp:Maze2DRandRecGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': RandMaze2DEnv,
            'num_groups': 12,
            'samples_per_env': 1000,
            'num_walls': 6,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.01?
                                    seed_planner=None,
                                    planner_timeout=30,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400),
            'eval_num_groups': 6,
            'eval_num_probs': 20,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 3,
            'dataset_url':f'{root_dir}/randSmaze2d-testOnly-v0.hdf5',
        }
    )


    hr = 0.70
    register(
        id='randSmaze2d-testOnly-v1',
        entry_point='pb_diff_envs.environment.static.maze2d_rand_wgrp:Maze2DRandRecGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': RandMaze2DEnv,
            'num_groups': 12,
            'samples_per_env': 1000,
            'num_walls': 3,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.01?
                                    seed_planner=None,
                                    planner_timeout=30,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400),
            'eval_num_groups': 6,
            'eval_num_probs': 20,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 3,
            'dataset_url':f'{root_dir}/randSmaze2d-testOnly-v1.hdf5',
        }
    )

    ## -------------------------------------------------
    ## ------------------ TestOnly ---------------------
    ## -------------------------------------------------

    hr = 1.0
    register(
        id='DynrandSmaze2d-testOnly-v2',
        entry_point='pb_diff_envs.environment.dynamic.dyn_rm2d_wgrp:DynRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': DynamicRandMaze2DEnv,
            'num_groups': 12,
            'samples_per_env': 200,
            'num_walls': 1,
            'maze_size': np.array([5,5]),
            'hExt_range': np.array([hr, hr]),
            'is_static': False,
            'num_sample_pts': 2,
            'speed': 0.12,
            'mazelist_config': dict(interp_density=None,
                                    gap_betw_wall=0.15,
                                    min_to_wall_dist=0.01,
                                    seed_planner=None,
                                    planner_timeout=300,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=500,
                                    robot_collision_eps=0.04,
                                    k_nb=50,),
            'eval_num_groups': 6,
            'eval_num_probs': 200, # not problems, len of samples, will be divided by horizon e.g. 48
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 6,
            'dataset_url':f'{root_dir}/DynrandSmaze2d-testOnly-v2.hdf5',
        }
    )




    ## static, composed
    nw1 = 6; nw2 = 3
    hr1 = 0.5; hr2 = 0.7; ct = 'direct'
    register(
        id='ComprandSmaze2d-testOnly-v3',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': 12,
            'n_comp': 2,
            'samples_per_env': 500,
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=False),
            'eval_num_groups': 6,
            'eval_num_probs': 20,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 6,
            'dataset_url':f'{root_dir}/ComprandSmaze2d-testOnly-v3.hdf5',
        }
    )

    # Mar 21, for testing, for multi wall loc mode, rebuttal icml 2024
    # composed 6 + 1
    nw1 = 6; nw2 = 1
    hr1 = 0.5; hr2 = 0.5; ct = 'share'
    register(
        id='ComprandSmaze2d-testOnly-nw61-0505-v0',
        entry_point='pb_diff_envs.environment.comp.comp_rand_m2d_wgrp:ComposedRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedRM2DEnv,
            'num_groups': 12, # not used
            'n_comp': 2,
            'samples_per_env': 500, # not used
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': True,
            'mazelist_config': dict(interp_density=8,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=20,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=400,
                                    grp_seed_diff=True,),
            'eval_num_groups': 6,
            'eval_num_probs': 20,
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 6, # not used
            'dataset_url':f'{root_dir}/ComprandSmaze2d-testOnly-nw61-0505-v0.hdf5',
        }
    )



    ## dynamic+staic, composed
    nw1 = 1; nw2 = 3
    hr1 = 1.0; hr2 = 0.7; ct = 'direct'
    register(
        id='CompDynrandSmaze2d-testOnly-v4',
        entry_point='pb_diff_envs.environment.comp.comp_stdyn_m2d_wgrp:ComposedStDynRM2DGroupList',
        max_episode_steps=150,
        kwargs={
            'robot_env': ComposedStDynRandMaze2DEnv,
            'num_groups': 12,
            'n_comp': 2,
            'samples_per_env': 200,
            'num_walls_c': np.array([nw1, nw2]),
            'maze_size': np.array([5,5]),
            'hExt_range_c': np.array([ [hr1, hr1], [hr2, hr2] ]),
            'comp_type': ct,
            'is_static': False,
            'num_sample_pts': 2,
            'speed': 0.12,
            'wall_is_dyn': [True, False, False, False],
            'mazelist_config': dict(interp_density=None,
                                    gap_betw_wall=0.15, # 0.15?
                                    min_to_wall_dist=0.01, # 0.02
                                    seed_planner=None,
                                    planner_timeout=300,
                                    rand_iter_limit=1e5,
                                    planner_num_batch=500, # 1000?
                                    robot_collision_eps=0.04, # ?
                                    k_nb=50,
                                    grp_seed_diff=False,
                                    ),
            'eval_num_groups': 6,
            'eval_num_probs': 100, # not problems, len of samples
            'seed': 333,
            'eval_seed_offset': 27,
            'gen_num_parts': 12,
            'dataset_url':f'{root_dir}/CompDynrandSmaze2d-testOnly-v4.hdf5',
        }
    )


  