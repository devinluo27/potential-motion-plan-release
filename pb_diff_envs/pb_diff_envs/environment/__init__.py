from gym.envs.registration import register
from pb_diff_envs.environment.static.rand_kuka_env import RandKukaEnv
from pb_diff_envs.environment.static.rand_dualkuka14d_env import RandDualKuka14dEnv
import numpy as np
from pb_diff_envs.environment.init_maze2d_gym import register_maze2d
from pb_diff_envs.environment.init_maze2d_gym_43 import register_maze2d_concave_43
from pb_diff_envs.environment.static.comp_kuka_env import ComposedRandKukaEnv, ComposedRandDualKuka14dEnv
import os, pdb

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '..', 'datasets')

eval_ng_300 = 300
eval_np_200 = 200
comp_eval_ng_100 = 100
comp_eval_np_20 = 40
comp_eval_np_40 = 40



## =============== kuka7d =================

hr = 0.20
vr_x = 0.2; vr_y = 0.2; vr_z = (0., 0.)
se_x = 0.5; se_y = 0.5; se_z = (0.3, 0.9)
register(
    id='kuka7d-ng2ks25k-rand-nv4-se0505-vr0202-hExt20-v0',
    entry_point='pb_diff_envs.environment.static.kuka_rand_vgrp:Kuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': RandKukaEnv,
        'num_groups': 2000, # 3
        'samples_per_env': 25000,
        'num_voxels': 4,
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range': np.array([hr, hr, hr]),
        'is_static': True,
        'mazelist_config': dict(interp_density=3,
                                seed_planner=None,
                                planner_timeout=30,
                                planner_num_batch=200),
        'eval_num_groups': eval_ng_300,
        'eval_num_probs': eval_np_200,
        'seed': 777,
        'eval_seed_offset': 27,
        'gen_num_parts': 500,
        'dataset_url':f'{root_dir}/kuka7d-ng2ks25k-rand-nv4-se0505-vr0202-hExt20-v0.hdf5',
    }
)





## =============== kuka14d =================
# avg traj len of one problem ~50
hr = 0.22
vr_x = 0.7; vr_y = 0.2; vr_z = (0., 0.)
se_x = 1.0; se_y = 0.5; se_z = (0.3, 0.9)
register(
    id='dualkuka14d-ng25hs25k-rand-nv5-se1005-vr0702-hExt22-v0',
    entry_point='pb_diff_envs.environment.static.dualkuka_rand_vgrp:DualKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': RandDualKuka14dEnv,
        'num_groups': 2500, # 15000, not work; 1500 work
        'samples_per_env': 25000,
        'num_voxels': 5,
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range': np.array([hr, hr, hr]),
        'is_static': True,
        'mazelist_config': dict(interp_density=1.5,
                                seed_planner=None,
                                planner_timeout=60,
                                rand_iter_limit=1e5,
                                planner_num_batch=200),
        'eval_num_groups': eval_ng_300, #
        'eval_num_probs': eval_np_200, #
        'seed': 666,
        'eval_seed_offset': 27,
        'gen_num_parts': 500,
        'dataset_url':f'{root_dir}/dualkuka14d-ng25hs25k-rand-nv5-se1005-vr0702-hExt22-v0.hdf5',
    }
)






#  ---------------------------------------------------------
## ------------ NOTE: Base TestOnly Template ---------------
#  ---------------------------------------------------------

hr = 0.20
vr_x = 0.2; vr_y = 0.2; vr_z = (0., 0.)
se_x = 0.5; se_y = 0.5; se_z = (0.3, 0.9)
register(
    id='kuka7d-testOnly-v8',
    entry_point='pb_diff_envs.environment.static.kuka_rand_vgrp:Kuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': RandKukaEnv,
        'num_groups': 20, # 3
        'samples_per_env': 500,
        'num_voxels': 4,
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range': np.array([hr, hr, hr]),
        'is_static': True,
        'mazelist_config': dict(interp_density=3,
                                seed_planner=None,
                                planner_timeout=30,
                                planner_num_batch=200),
        'eval_num_groups': 10,
        'eval_num_probs': 6,
        'seed': 777,
        'eval_seed_offset': 27,
        'gen_num_parts': 5,
        'dataset_url':f'{root_dir}/kuka7d-testOnly-v8.hdf5',
    }
)



## Testing Env
## =============== dualkuka14d =================
hr = 0.20
vr_x = 0.7; vr_y = 0.2; vr_z = (0., 0.)
se_x = 1.0; se_y = 0.5; se_z = (0.3, 0.9)
register(
    id='dualkuka14d-testOnly-v9',
    entry_point='pb_diff_envs.environment.static.dualkuka_rand_vgrp:DualKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': RandDualKuka14dEnv,
        'num_groups': 12,
        'samples_per_env': 200,
        'num_voxels': 5,
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,

        'orn_range': None,
        'hExt_range': np.array([hr, hr, hr]),
        'is_static': True,
        'mazelist_config': dict(interp_density=1.5,
                                seed_planner=None,
                                planner_timeout=60),
        'eval_num_groups': 10, # eval_ng_300,
        'eval_num_probs': 6, # eval_np_200,
        'seed': 333,
        'eval_seed_offset': 27,
        'gen_num_parts': 5,
        'dataset_url':f'{root_dir}/dualkuka14d-testOnly-v9.hdf5',
    }
)



## --------------------------------------------------------
## -------------- NOTE: Composed TestOnly -----------------
## --------------------------------------------------------

## kuka 7D, static, composed
hr1 = hr2 = 0.20; ct = 'share'
vr_x = 0.2; vr_y = 0.2; vr_z = (0., 0.)
se_x = 0.5; se_y = 0.5; se_z = (0.3, 0.9)
nw1 = 4; nw2 = 2
register(
    id='Compkuka7d-testOnly-rand-v11',
    entry_point='pb_diff_envs.environment.comp.comp_rand_kuka_vgrp:ComposedKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': ComposedRandKukaEnv,
        'num_groups': 9,
        'n_comp': 2,
        'samples_per_env': 200,
        'num_walls_c': np.array([nw1, nw2]),
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range_c': np.array([ [hr1, hr1, hr1], [hr2, hr2, hr2] ]),
        'comp_type': ct,
        'is_static': True,

        'mazelist_config': dict(interp_density=3,
                                ##
                                seed_planner=None,
                                planner_timeout=30,
                                rand_iter_limit=1e5,
                                planner_num_batch=400,
                                grp_seed_diff=True),
        'eval_num_groups': 6,
        'eval_num_probs': 5,
        'seed': 333,
        'eval_seed_offset': 27,
        'gen_num_parts': 3,
        'dataset_url':f'{root_dir}/Compkuka7d-testOnly-rand-v11.hdf5',
    }
)



## dual kuka 14D, static, composed
hr1 = hr2 = 0.22; ct = 'share'
vr_x = 0.7; vr_y = 0.2; vr_z = (0., 0.)
se_x = 1.0; se_y = 0.5; se_z = (0.3, 0.9)
nw1 = 5; nw2 = 2
register(
    id='Compdualkuka14d-testOnly-rand-v12',
    entry_point='pb_diff_envs.environment.comp.comp_rand_kuka_vgrp:ComposedDualKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': ComposedRandDualKuka14dEnv,
        'num_groups': 2500, # not used
        'n_comp': 2,
        'samples_per_env': 25000, # not used
        'num_walls_c': np.array([nw1, nw2]),
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range_c': np.array([ [hr1, hr1, hr1], [hr2, hr2, hr2] ]),
        'comp_type': ct,
        'is_static': True,

        'mazelist_config': dict(interp_density=1.5,
                                seed_planner=None,
                                planner_timeout=60,
                                rand_iter_limit=1e5,
                                planner_num_batch=200, # or 400 same as 7d?
                                grp_seed_diff=True),
        'eval_num_groups': 6, #
        'eval_num_probs': 5, #
        'seed': 666,
        'eval_seed_offset': 27,
        'gen_num_parts': 3,
        'dataset_url':f'{root_dir}/Compdualkuka14d-testOnly-rand-v12.hdf5',
    }
)











# ------------------------------------------------------
# -------------- NOTE: composed Kuka 7D ----------------
# ------------------------------------------------------


## kuka 7D, static, composed 4 + 1
hr1 = hr2 = 0.20; ct = 'share'
vr_x = 0.2; vr_y = 0.2; vr_z = (0., 0.)
se_x = 0.5; se_y = 0.5; se_z = (0.3, 0.9)
nw1 = 4; nw2 = 1
register(
    id='Compkuka7d-rand-nv4nv1-se0505-vr0202-hExt20-v0',
    entry_point='pb_diff_envs.environment.comp.comp_rand_kuka_vgrp:ComposedKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': ComposedRandKukaEnv,
        'num_groups': 2000, # not used
        'n_comp': 2,
        'samples_per_env': 25000, # not used
        'num_walls_c': np.array([nw1, nw2]),
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range_c': np.array([ [hr1, hr1, hr1], [hr2, hr2, hr2] ]),
        'comp_type': ct,
        'is_static': True,

        'mazelist_config': dict(interp_density=3,
                                ##
                                seed_planner=None,
                                planner_timeout=30,
                                rand_iter_limit=1e5,
                                planner_num_batch=400,
                                grp_seed_diff=True),
        'eval_num_groups': comp_eval_ng_100,
        'eval_num_probs': comp_eval_np_40,
        'seed': 333,
        'eval_seed_offset': 27,
        'gen_num_parts': 3,
        'dataset_url':f'{root_dir}/Compkuka7d-rand-nv4nv1-se0505-vr0202-hExt20-v0.hdf5',
    }
)


## kuka 7D, static, composed 4 + 2
hr1 = hr2 = 0.20; ct = 'share'
vr_x = 0.2; vr_y = 0.2; vr_z = (0., 0.)
se_x = 0.5; se_y = 0.5; se_z = (0.3, 0.9)
nw1 = 4; nw2 = 2
register(
    id='Compkuka7d-rand-nv4nv2-se0505-vr0202-hExt20-v0',
    entry_point='pb_diff_envs.environment.comp.comp_rand_kuka_vgrp:ComposedKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': ComposedRandKukaEnv,
        'num_groups': 2000, # not used
        'n_comp': 2,
        'samples_per_env': 25000, # not used
        'num_walls_c': np.array([nw1, nw2]),
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range_c': np.array([ [hr1, hr1, hr1], [hr2, hr2, hr2] ]),
        'comp_type': ct,
        'is_static': True,

        'mazelist_config': dict(interp_density=3,
                                ##
                                seed_planner=None,
                                planner_timeout=30,
                                rand_iter_limit=1e5,
                                planner_num_batch=400,
                                grp_seed_diff=True),
        'eval_num_groups': comp_eval_ng_100,
        'eval_num_probs': comp_eval_np_40,
        'seed': 333,
        'eval_seed_offset': 27,
        'gen_num_parts': 3,
        'dataset_url':f'{root_dir}/Compkuka7d-rand-nv4nv2-se0505-vr0202-hExt20-v0.hdf5',
    }
)



## kuka 7D, static, composed 4 + 3
hr1 = hr2 = 0.20; ct = 'share'
vr_x = 0.2; vr_y = 0.2; vr_z = (0., 0.)
se_x = 0.5; se_y = 0.5; se_z = (0.3, 0.9)
nw1 = 4; nw2 = 3
register(
    id='Compkuka7d-rand-nv4nv3-se0505-vr0202-hExt20-v0',
    entry_point='pb_diff_envs.environment.comp.comp_rand_kuka_vgrp:ComposedKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': ComposedRandKukaEnv,
        'num_groups': 2000, # not used
        'n_comp': 2,
        'samples_per_env': 25000, # not used
        'num_walls_c': np.array([nw1, nw2]),
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range_c': np.array([ [hr1, hr1, hr1], [hr2, hr2, hr2] ]),
        'comp_type': ct,
        'is_static': True,

        'mazelist_config': dict(interp_density=3,
                                ##
                                seed_planner=None,
                                planner_timeout=30,
                                rand_iter_limit=1e5,
                                planner_num_batch=400,
                                grp_seed_diff=True),
        'eval_num_groups': comp_eval_ng_100,
        'eval_num_probs': comp_eval_np_40,
        'seed': 333,
        'eval_seed_offset': 27,
        'gen_num_parts': 3,
        'dataset_url':f'{root_dir}/Compkuka7d-rand-nv4nv3-se0505-vr0202-hExt20-v0.hdf5',
    }
)







# ------------------------------------------------
# -------- NOTE: composed Dualkuka 14D -----------
# ------------------------------------------------



# dual 5 + 1
hr1 = hr2 = 0.22; ct = 'share'
vr_x = 0.7; vr_y = 0.2; vr_z = (0., 0.)
se_x = 1.0; se_y = 0.5; se_z = (0.3, 0.9)
nw1 = 5; nw2 = 1
register(
    id='Compdualkuka14d-rand-nv5nv1-se1005-vr0702-hExt22-v0',
    entry_point='pb_diff_envs.environment.comp.comp_rand_kuka_vgrp:ComposedDualKuka_VoxelRandGroupList',
    max_episode_steps=150,
    kwargs={
        'robot_env': ComposedRandDualKuka14dEnv,
        'num_groups': 2500, # not used
        'n_comp': 2,
        'samples_per_env': 25000, # not used
        'num_walls_c': np.array([nw1, nw2]),
        'start_end': np.array([[-se_x, se_x], [-se_y, se_y], se_z]),
        'void_range': np.array( [(-vr_x, vr_x), (-vr_y, vr_y), vr_z] ) ,
        'orn_range': None,
        'hExt_range_c': np.array([ [hr1, hr1, hr1], [hr2, hr2, hr2] ]),
        'comp_type': ct,
        'is_static': True,

        'mazelist_config': dict(interp_density=1.5,
                                seed_planner=None,
                                planner_timeout=60,
                                rand_iter_limit=1e5,
                                planner_num_batch=200, # or 400 same as 7d?
                                grp_seed_diff=True),
        'eval_num_groups': comp_eval_ng_100,
        'eval_num_probs': comp_eval_np_40,
        'seed': 666,
        'eval_seed_offset': 27,
        'gen_num_parts': 3,
        'dataset_url':f'{root_dir}/Compdualkuka14d-rand-nv5nv1-se1005-vr0702-hExt22-v0.hdf5',
    }
)








register_maze2d()
register_maze2d_concave_43()