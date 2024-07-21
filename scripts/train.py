import sys
sys.path.append('.')
import wandb
import diffuser.utils as utils
import pdb
import os.path as osp
import torch
torch.backends.cudnn.benchmark = True

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = None
    config: str = 'config/rm2d/rSmaze_engy_testOnly.py'

## **set dataset in parse_args automatically
args = Parser().parse_args('diffusion')

torch.backends.cudnn.allow_tf32 = getattr(args, 'allow_tf32', True)
torch.backends.cuda.matmul.allow_tf32 = getattr(args, 'allow_tf32', True)

print('args.dataset', args.dataset)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    max_n_episodes=getattr(args, 'max_n_episodes', 10000),

    dataset_config=getattr(args, 'dataset_config', {}),
    termination_penalty=args.termination_penalty, # useless, not pass

)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset, # can input the env instance here to save time.
    is_sol_kp_mode=getattr(args, 'is_sol_kp_mode', False),
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim, # keydiff
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    dim=getattr(args, 'dim', 32), # 32 is the default
    device=args.device,
    

    num_walls=args.num_walls,
    wall_embed_dim=args.wall_embed_dim,
    wall_sinPosEmb=getattr(args, 'wall_sinPosEmb', None), ## should be a dict
    network_config=getattr(args, 'network_config', {}) ## var_temp=0.5 is better
)


diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath,'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type, ## checkparam
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,

    diff_config=getattr(args, 'diff_config', {}),
    
    ## loss weighting
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    
    condition_guidance_w=args.condition_guidance_w,
    device=args.device,
)

trainer_dict = getattr(args, 'trainer_dict', {})

trainer_config = utils.Config(
    utils.TrainerPBDiff,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    bucket=args.bucket,
    n_reference=args.n_reference,


    ## -------------------------
    log_freq=args.log_freq,
    train_device=args.device,
    save_checkpoints=args.save_checkpoints,

    results_folder=args.savepath,
    n_samples=args.n_samples,
    ## -------------------------
    num_render_samples=getattr(args, "num_render_samples", 2),
    clip_grad=getattr(args, 'clip_grad', False),
    clip_grad_max=getattr(args, 'clip_grad_max', False),
    n_train_steps=args.n_train_steps,
    trainer_dict=trainer_dict,
    
    step_start_ema=trainer_dict.get('step_start_ema', 2000),
    update_ema_every=trainer_dict.get('update_ema_every', 10),
)


#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

print('args.diffusion:', args.diffusion)

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

if trainer_dict.get('do_train_resume', False): # for a sample resume, should be good
    trainer.load4resume( trainer_dict['path_resume'] )


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])

loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- save config ---------------------------------#
#-----------------------------------------------------------------------------#


all_configs = dict(dataset_config=dataset_config._dict, 
                render_config=render_config._dict,
                model_config=model_config._dict,
                diffusion_config=model_config._dict,
                trainer_config=trainer_config._dict)
# print(args)
ckp_path = args.savepath
wandb.init(
    project="Potential-Motion-Plan-Release",
    name=args.logger_name,
    id=args.logger_id,
    dir=ckp_path,
    config=all_configs, ## need to be a dict
    # resume="must",
)

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

print("Finsied Training")
exit(0)
