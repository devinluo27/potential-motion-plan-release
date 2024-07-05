import os
import copy
import numpy as np
import torch
import einops

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .train_utils import cycle, EMA
from diffuser.utils.train_utils import get_lr
import diffuser.models as dmodels
from diffuser.utils.rendering import KukaRenderer
from diffuser.utils.rm2d_render import RandStaticMazeRenderer, RandDynamicMazeRenderer
import wandb
from diffuser.utils.train_utils import CosineAnnealingWarmupRestarts
from tqdm import tqdm


class TrainerPBDiff(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        results_folder='./results',
        n_samples=2, ## sample times per traj
        num_render_samples=2, ## 
        clip_grad=False,
        clip_grad_max=False,
        n_train_steps=None,
        trainer_dict={},
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema

        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=8, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder

        self.bucket = results_folder
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.num_render_samples = num_render_samples
        self.clip_grad = clip_grad # grad norm 
        self.clip_grad_max = clip_grad_max # max grad
        self.lr_warmupDecay = trainer_dict.get('lr_warmupDecay', False)
        self.is_kuka = trainer_dict.get('is_kuka', False)
        self.is_dyn_env = (type(self.renderer) == RandDynamicMazeRenderer)
        if self.lr_warmupDecay:
            assert n_train_steps
            warmup_steps = trainer_dict['warmup_steps_pct'] * n_train_steps
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                          first_cycle_steps=n_train_steps,
                                          max_lr=train_lr,
                                          min_lr=train_lr/100.,
                                          warmup_steps=warmup_steps,)

        self.reset_parameters()
        self.step = 0
        self.debug_mode = False

        self.device = train_device


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)

                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                
                loss.backward()

                ## gradient clipping
                if self.clip_grad_max:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_max)
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)


            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_warmupDecay:
                self.scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            ## checkdesign
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                # a0 loss is from self.model
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])

                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

                metrics = {k:v.detach().item() for k, v in infos.items()}
                
                metrics['train/it'] = self.step
                metrics['train/loss'] = loss.detach().item()
                metrics['train/lr'] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)
                
            if self.step == 0 and self.sample_freq:
                # ref_bs = 10 if not self.is_kuka else 1
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.model.__class__ == dmodels.GaussianDiffusionPB:
                    self.inv_render_samples(batch_size=self.num_render_samples, n_samples=self.n_samples,)
                else:
                    raise NotImplementedError
                    

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }

        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        # loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
    
    def load4resume(self, loadpath):
        data = torch.load(loadpath)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)



        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')


        if type(self.renderer) == KukaRenderer or \
                type(self.renderer) == RandStaticMazeRenderer:
            ## batch.maze_idx torch.Size([50, 1])
            # print('batch.maze_idx', batch.maze_idx.shape)
            self.renderer.composite(savepath, observations, maze_idx=batch.maze_idx)
        elif type(self.renderer) == RandDynamicMazeRenderer:

            wtrajs_list = batch.wall_locations # tensor B, 48, nw*dim
            self.renderer.composite(savepath, observations, batch.maze_idx, wtrajs_list)
        else:
            raise NotImplementedError

    

    def inv_render_samples(self, batch_size=2, n_samples=2, vis_loss=False):
        '''
            this func is for using invdynGaussian
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            # batch[0] torch.Size([1, 240, 4])

            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            model_type = type(self.ema_model)
            is_training = self.ema_model.training # get status
            self.ema_model.eval() # set to eval mode


            if model_type == dmodels.GaussianDiffusionPB:
                ## wall_locations torch.Size([1, h, 6])
                if self.is_dyn_env:
                    wloc = torch.flatten(batch.wall_locations, start_dim=1, end_dim=2)
                else:
                    wloc = batch.wall_locations[:,0,:]
                walls_loc = einops.repeat(wloc, 'b d -> (repeat b) d',repeat=n_samples)
                samples = self.ema_model.conditional_sample(conditions, walls_loc,)
            else:
                raise NotImplementedError()
            

            self.ema_model.train(is_training) # restore to original mode

            samples = to_np(samples)

            ## checkdesign 
            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]
            normed_goal = to_np(batch.conditions[self.model.horizon-1])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                ## for rendering, insert start point
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations,
                ## insert end point
                np.repeat(normed_goal, n_samples, axis=0),
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')


            sample_savedir = self.get_sample_savedir(self.step)

            if self.debug_mode:
                sample_savedir = os.path.join(self.logdir, f'debug-vis')
                if not os.path.isdir(sample_savedir):
                    os.makedirs(sample_savedir)

            savepath = os.path.join(sample_savedir, f'sample-{self.step}-{i}.png')
            if model_type == dmodels.GaussianDiffusionPB:

                ## Before: batch.maze_idx torch.Size([1, 1]) -> (n,1)
                maze_idx = einops.repeat(
                    ## b is 1
                    batch.maze_idx, 'b d -> (repeat b) d',repeat=n_samples)
                ## batch.maze_idx torch.Size([1, 1]) (10, 145(+1), 4)
                print('batch.maze_idx', batch.maze_idx.shape, 'obs', observations.shape)
                ## added for dynamic env
                if self.is_dyn_env:
                    wloc = vis_preproc_dyn_wtraj(batch, n_samples)
                    self.renderer.composite(savepath, observations, maze_idx=maze_idx, wtrajs_list=wloc)
                else:
                    self.renderer.composite(savepath, observations, maze_idx=maze_idx)

            else: 
                raise NotImplementedError
    


    def get_sample_savedir(self, i):
        div_freq = 50000
        subdir = str( (i // div_freq) * div_freq )
        sample_savedir = os.path.join(self.logdir, subdir)
        if not os.path.isdir(sample_savedir):
            os.makedirs(sample_savedir)
        return sample_savedir



def vis_preproc_dyn_wtraj(batch, n_samples):
    '''pad wtrajs for dynamic env'''
    wloc = einops.repeat(
                    batch.wall_locations, 'b h d -> (repeat b) h d',repeat=n_samples)

    wloc = np.concatenate([
            wloc[:, 0:1],
            wloc,
            wloc[:, -1:]
        ], axis=1)

    return wloc