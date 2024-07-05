 
import os.path as osp

from diffuser.utils import watch
from diffuser.datasets import DUALKUKA14D_MIN, DUALKUKA14D_MAX

## dualk14Inv_Lng25hs25k_nw5hExt22_vit_h48_engyL2_bs256ac1_muld64_nodp_tmlp3
#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args
config_fn = osp.splitext(osp.basename(__file__))[0]
diffusion_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

## check 1.input_size,2.embed_dim,3.attn_dim
## embed_dim is the dim to encoder each wallLoc(2,), which is fed to transformer
vit_emb = 64
num_walls = 5
vit1d_config = dict(
    input_size=num_walls*3, patch_size=3, num_classes=0, 
    embed_dim=vit_emb, depth=3, mlp_ratio=4, pool='cls',
    attn_dim=vit_emb, num_heads=1, dim_head=vit_emb, channels=1, dropout=0.0, emb_dropout = 0.0, tf_config={'mish': False}
)
norm_const_dict = {'observations': (DUALKUKA14D_MIN, DUALKUKA14D_MAX)}
base = {
    # 'dataset': "kuka7d-luotest-rand14d-v9",
    'dataset': "dualkuka14d-ng25hs25k-rand-nv5-se1005-vr0702-hExt22-v0",
    'diffusion': {
        'config_fn': '',
        ## model
        'model': 'models.TemporalUnet_WCond',
        'diffusion': 'models.GaussianDiffusionPB',
        'horizon': 48,
        'n_diffusion_steps': 100,  
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': True,
        'dim': 64,
        'dim_mults': (1, 4, 8, 12), # no down last
        'num_walls': num_walls, # only used when mlp
        # 'var_temp': 0.5,
        'wall_embed_dim': vit1d_config['embed_dim'], ## used only in MLP cases
        'network_config': dict(wallLoc_encoder='vit1d', vit1d_config=vit1d_config,
                               concept_drop_prob=0.2, cat_t_w=True,
                               energy_mode=True,
                               energy_param_type='L2',
                               conv_zero_init=False,
                               last_conv_ksize=1,
                               time_mlp_config=3,
                               down_times=2, ##
                               ),

        'diff_config': dict(set_cond_noise_to_0=False, 
                            manual_loss_weights={0:0.0, -1:0.0}),
        'trainer_dict': dict(is_kuka=True,),
        'allow_tf32': True,
        
        'renderer': 'utils.KukaRenderer',

        ## checkparam new in dd
        'hidden_dim': -1, # dim in inv,
        'ar_inv': False,
        'train_only_inv': False,
        'condition_guidance_w': 2.0,
        ## NotUsed
        'returns_condition': False,
        'condition_dropout': -1.0,
        'calc_energy': False,



        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'dataset_config': dict(is_mazelist=True, 
                               use_normed_wallLoc=False,
                               obs_selected_dim=tuple(range(14)),
                               maze_change_as_terminal=True,
                               norm_const_dict=norm_const_dict,),


        'preprocess_fns': ['maze2d_set_terminals_luo'],
        'clip_denoised': True,
        'use_padding': False,


        'max_path_length': 25005,
        'max_n_episodes': 2700,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 256,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 3000, # ori:1000
        'sample_freq': 5000, # ori: 1000
        'n_saves': 20, # num of checkpoint to save
        'num_render_samples': 2, ##
        'save_parallel': False,
        'n_reference': 2,
        'n_samples': 1,
        'bucket': None,
        'device': 'cuda',
        ## checkparam new in dd
        'log_freq': 100,
        'save_checkpoints': False,

    },

    'plan': {
        'config_fn': '',
        'load_unseen_mazelist': True,

        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 48,
        'n_diffusion_steps': 100,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': True,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest', #

    },

}
