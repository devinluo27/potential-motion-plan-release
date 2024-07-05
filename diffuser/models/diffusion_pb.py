import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    set_loss_noise,
    Losses,
)
import matplotlib.pyplot as plt
from diffuser.utils.luo_utils import plot_xy, batch_repeat_tensor
from colorama import Fore

class GaussianDiffusionPB(nn.Module):
    '''
    Diffusion Module
    Only Predict a Trajectory of joint state or position
    '''
    def __init__(self, 
                model, 
                horizon, 
                observation_dim, 
                action_dim, 
                n_timesteps=1000,
                loss_type='l2', 
                clip_denoised=False, 
                predict_epsilon=True, 
                loss_discount=1.0, 
                loss_weights=None, 
                condition_guidance_w=0.1, 
                diff_config={}
        ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model


        ## we need grad if in energy mode
        self.energy_mode = self.model.energy_mode # not ok
        
        self.condition_guidance_w = condition_guidance_w
        self.train_apply_condition = diff_config.get('train_apply_condition', True)

        # set to False to be Aligned with previous training
        self.set_cond_noise_to_0 = diff_config.get('set_cond_noise_to_0', False)
        print(f'[Luo self.train_apply_condition] {self.train_apply_condition}')
        print(f'[Luo self.set_cond_noise_to_0] {self.set_cond_noise_to_0}')
        self.debug_mode = diff_config.get('debug_mode', False)
        self.manual_loss_weights = diff_config.get('manual_loss_weights', {})
        ## 1.0 for energy_mode
        self.var_temp = 1.0 if self.energy_mode else 0.5 

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        ## ------ setup ddim --------
        self.num_train_timesteps = self.n_timesteps

        self.ddim_num_inference_steps = diff_config.get('ddim_steps', 10)


        ddim_set_alpha_to_one = diff_config.get('ddim_set_alpha_to_one', True)
        self.final_alpha_cumprod = torch.tensor([1.0,], ) \
            if ddim_set_alpha_to_one else torch.clone(self.alphas_cumprod[0:1]) # tensor of size (1,)

        ## ---------------------------
        self.is_dyn_env = diff_config.get('is_dyn_env', False)

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        assert loss_type == 'l2'
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0
        if len(self.manual_loss_weights) > 0:
            for k, v in self.manual_loss_weights.items():
                loss_weights[k, :] = v
                print(f'[set manual loss weight] {k} {v}')

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, walls_loc, return_modelout=False):
        '''use in inference cond is acutally not used'''
        
        
        x = x.detach()
        
        ## get cond and uncond in one forward
        x_2, cond_2, t_2, walls_loc_2 = batch_repeat_tensor(x, cond, t, walls_loc, 2)
        x_2 = x_2.detach()
        out = self.model(x_2, cond_2, t_2, walls_loc_2, use_dropout=False, force_dropout=True, half_fd=True)

        epsilon_cond = out[:len(t), :, :]
        epsilon_uncond = out[len(t):, :, :]
        assert epsilon_cond.shape == epsilon_uncond.shape


        epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)


        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            ## must do clamp
            x_recon.clamp_(-1., 1.)

        else:
            raise RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        if return_modelout:
            return model_mean, posterior_variance, posterior_log_variance, x_recon, epsilon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def pred_x_tm1_luo(self, x, t, epsilon, clip_denoised):
        '''
        Usage: 
        1. use `pred_unet_luo` to get epsilon; 2. input the combine epsilon to `pred_x_tm1_luo`
        [Added by Luo. Follow original code.]
        Only different is that now epsilon is given rather than computed in the func.
        pred x_{t-1} given x_t, t, and epsilon.
        Args:
            x (B, horizon, dim): x_t in equation (e.g.m DDPM eq.15)
            t (B,):
            epsilon (B, horizon, dim):
        '''
        assert x.shape[-1] == self.observation_dim

        ## get x_0 from DDPM eq.15
        x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * epsilon
        
        
        if self.clip_denoised and clip_denoised:

            x_recon.clamp_(-1., 1.)
        else:
            utils.print_color('x_recon not clip')

        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)

        noise = self.var_temp * torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        ## sample a value: input x_t output x_{t-1}
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise



    def p_sample(self, x, cond, t, walls_loc,):
        '''t (cuda tensor [B,])'''
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, walls_loc=walls_loc,)
        # noise = 0.5*torch.randn_like(x)
        noise = self.var_temp * torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    def p_sample_loop(self, shape, cond, walls_loc, verbose=True, return_diffusion=False):


        device = self.betas.device

        batch_size = shape[0]

        x = self.var_temp*torch.randn(shape, device=device)

        x = apply_conditioning(x, cond, 0) # start from dim 0, different from d

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            walls_loc = walls_loc.to(device)
            x = self.p_sample(x, cond, timesteps, walls_loc,)
            
            x = apply_conditioning(x, cond, 0)

            # progress.update({'t': i})

            x = x.detach()

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def replan(self, traj, cond, walls_loc, dn_steps):
        '''dn_steps: a int, noisy level'''
        device = self.betas.device
        batch_size = 1

        assert torch.is_tensor(traj)

        input2d = traj.ndim < 3
        if input2d:
            traj = traj[None, ] # [1, h, 7])

        ## 1. add 20 steps of noise
        assert dn_steps < self.n_timesteps and dn_steps > self.n_timesteps / 11
        # pdb.set_trace()
        noisy_traj = self.q_sample(traj, dn_steps)
        x = apply_conditioning(noisy_traj, cond, 0) # use x, consistent with other method

        ## 2. denoise
        for i in reversed(range(0, dn_steps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, walls_loc, None)
            x = apply_conditioning(x, cond, 0)
            x = x.detach()
            
        if input2d:
            x = x[0]
        return x
    
    def ddim_replan(self, traj, cond, walls_loc, dn_steps: torch.Tensor, ret_noisy_traj=False):
        '''
        dn_steps (int): refer to number of ddim denoising steps to run
        ret_noise_traj: only return a noisy traj
        '''
        # utils.print_color(f'ddim replan steps: {dn_steps}')
        device = self.betas.device
        batch_size = 1
        assert torch.is_tensor(traj)

        input2d = traj.ndim < 3
        if input2d:
            traj = traj[None, ] # [1, h, 7])

        ## 1. add 20 steps of noise
        assert dn_steps <= self.ddim_num_inference_steps * 0.51
        # e.g., [90, 80, ..., 20, 10, 0]
        ts = self.ddim_set_timesteps(self.ddim_num_inference_steps) # np

        noise_t = ts[-dn_steps].item()
        noise_t = torch.tensor([noise_t,], device=device)
        noisy_traj = self.q_sample(traj, noise_t) # noise_t must be a tensor [B,]
        if ret_noisy_traj: # for policy compose: [1, h, dim]
            return noisy_traj
        x = apply_conditioning(noisy_traj, cond, 0) # use x, consistent with other method

        ## 2. ddim denoise
        for i in ts[-dn_steps:]:
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.ddim_p_sample(x, cond, timesteps, walls_loc, eta=0.0, use_clipped_model_output=True)
            
            x = apply_conditioning(x, cond, 0)
            x = x.detach()
            
        if input2d:
            x = x[0]
        return x


    def ddim_get_noisy_traj(self, traj, inference_steps, dn_steps: torch.Tensor, check_steps=True):
        '''
        dn_steps (int): refer to number of ddim denoising steps to run
        ret_noise_traj: only return a noisy traj
        '''
        device = self.betas.device
        assert torch.is_tensor(traj)

        input2d = traj.ndim < 3
        if input2d:
            traj = traj[None, ] # [1, h, 7])

        if check_steps:
            ## 1. add 20 steps of noise
            assert dn_steps < inference_steps * 0.5
        # e.g., [90, 80, ..., 20, 10, 0]
        ts = self.ddim_set_timesteps(inference_steps) # np

        noise_t = ts[-dn_steps].item()
        noise_t = torch.tensor([noise_t,], device=device)
        noisy_traj = self.q_sample(traj, noise_t) # noise_t must be a tensor [B,]
        return noisy_traj
    


    def pred_unet(self, x_t, cond, t, walls_loc, return_epsilon=True, mala_sampler=False):
        '''
        given (x_t, cond, t) get the output epsilon (noise), 
        1. cond, 2. uncond
        can be called directly from outside, used for composing
        Args:
            x_t: noisy trajectories
            t (int or cuda tensor): a reversed timestep, should be (n_timesteps - 1) -> 0
        Returns:
            two tuples
        1. if t = 0: x should be a pure noise
        '''
        assert not self.training
        assert x_t.shape[-1] == self.observation_dim
        # x_t shape: (B, h, dim)

        ## ---- Integrate code from p_sample_loop, p_sample, ... --------
        device = self.betas.device
        shape = x_t.shape
        # print('x_t:', shape, 'walls_loc', walls_loc.shape)
        
        batch_size = shape[0]
        assert len(cond[0]) == batch_size

        x_t = apply_conditioning(x_t, cond, 0)
        if type(t) == int:
            t = torch.full((batch_size,), t, device=device, dtype=torch.long)
        else:
            assert t.shape == (batch_size,)


        walls_loc = walls_loc.to(device)
        
        ## [NOTE] out is either a noise or x_0
        with torch.set_grad_enabled(self.energy_mode):
            x_t.detach_()
            
            x_2, cond_2, t_2, walls_loc_2 = batch_repeat_tensor(x_t, cond, t, walls_loc, 2)
            x_2 = x_2.detach()
            out = self.model(x_2, cond_2, t_2, walls_loc_2, use_dropout=False, force_dropout=True, half_fd=True)

            if mala_sampler:
                engy = out[0].detach()
                out = out[1]
                assert engy.ndim == 1 and engy.shape[0] == out.shape[0]
            else:
                engy = [None,] * len(out)
            
            out = out.detach()
            epsilon_cond = out[:len(t), :, :]
            epsilon_uncond = out[len(t):, :, :]
            engy_cond = engy[ :len(t) ]
            engy_uncond = engy[ len(t): ]

            assert epsilon_cond.shape == epsilon_uncond.shape

        ## check if satisfy
        if return_epsilon and not self.predict_epsilon:
            ## convert x_0 to pred noise
            raise NotImplementedError()
        elif not return_epsilon:
            raise NotImplementedError()


        return (engy_cond, epsilon_cond),  (engy_uncond ,epsilon_uncond)




    def conditional_sample(self, cond, walls_loc, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        assert horizon is None
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        print('conditional_sample shape:', shape, 'walls_loc', walls_loc.shape)

        for k,v in cond.items():
            print(k, v.shape)

        use_ddim = kwargs.get('use_ddim', False)
        kwargs.pop('use_ddim', None)
        with torch.set_grad_enabled(self.energy_mode):
            if use_ddim:
                return self.ddim_p_sample_loop(shape, cond, walls_loc, *args, **kwargs)
            else:
                return self.p_sample_loop(shape, cond, walls_loc, *args, **kwargs)
    

    




    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, walls_loc, return_x_recon=False):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.train_apply_condition: # should be True
            x_noisy = apply_conditioning(x_noisy, cond, 0)
            if self.set_cond_noise_to_0:
                set_loss_noise(noise, cond, 0, 0.0)
        else:
            raise RuntimeError


        if self.energy_mode and self.training:
            x_recon, engy_batch = self.model(x_noisy, cond, t, walls_loc,)
        else:
            x_recon = self.model(x_noisy, cond, t, walls_loc,)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape, f'{noise.shape}, {x_recon.shape}'

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        if self.energy_mode and self.training:
            info['engy_batch'] = engy_batch
        if return_x_recon:
            info['x_start'] = self.predict_start_from_noise(x_noisy, t, x_recon)

        return loss, info

    def loss(self, x, cond, wall_locations, maze_idx=None): # placeholder
        '''
        is directly called from trainer.
        args:
            x: check what is x? x is the gt traj?
            cond (dict): from dataset cond[0]: B, obs_dim
        '''

        batch_size = len(x)

        if self.is_dyn_env: # tensor
            # NOTE B, 48, nw*dim -> B, 48*nw*dim 
            wall_locations = torch.flatten(wall_locations, start_dim=1, end_dim=2)
        else:
            wall_locations = wall_locations[:, 0, :]
        
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        ## NOTE input to p_losses is the state
        diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], 
                                            cond, t, wall_locations,)
        
        loss = (1 / 2) * diffuse_loss

        info['diffuse_loss'] = diffuse_loss.detach()

        return loss, info

    def forward(self, cond, walls_loc, *args, **kwargs):
        
        return self.conditional_sample(cond=cond, walls_loc=walls_loc, *args, **kwargs)


    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    


    def ddim_p_sample(self, x, cond, t, walls_loc, eta=0.0, use_clipped_model_output=False):
        ''' NOTE follow diffusers ddim, any post-processing *NOT CHECKED yet*
        t (cuda tensor [B,]) must be same
        eta: weight for noise'''

        # # 1. get previous step value (=t-1), (B,)
        prev_timestep = t - self.num_train_timesteps // self.ddim_num_inference_steps
        # # 2. compute alphas, betas
        alpha_prod_t = extract(self.alphas_cumprod, t, x.shape) # 
        if prev_timestep[0] >= 0:
            alpha_prod_t_prev = extract(self.alphas_cumprod, prev_timestep, x.shape) # tensor 
        else:
            # extract from a tensor of size 1, cuda tensor [80, 1, 1]
            alpha_prod_t_prev = extract(self.final_alpha_cumprod.to(t.device), torch.zeros_like(t), x.shape)
            # print(f'alpha_prod_t_prev {alpha_prod_t_prev[0:3]}')
        assert alpha_prod_t.shape == alpha_prod_t_prev.shape

        beta_prod_t = 1 - alpha_prod_t

        b, *_, device = *x.shape, x.device
        

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # 4. Clip "predicted x_0"
        ## model_mean is clipped x_0, 
        ## model_output: model prediction, should be the epsilon (noise)
        model_mean, _, model_log_variance, x_recon, model_output = self.p_mean_variance(x=x, cond=cond, t=t, walls_loc=walls_loc, return_modelout=True)


        ## 5. compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) \
            * ( 1 - alpha_prod_t / alpha_prod_t_prev )

        std_dev_t = eta * variance ** (0.5)

        assert use_clipped_model_output
        if use_clipped_model_output:

            sample = x
            pred_original_sample = x_recon
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            


        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        sample = prev_sample
        
        
        return sample
        


    def ddim_set_timesteps(self, num_inference_steps) -> np.ndarray: 

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        # e.g., 10: [90, 80, 70, 60, 50, 40, 30, 20, 10,  0]
        
        return timesteps


    def ddim_p_sample_loop(self, shape, cond, walls_loc, verbose=True, return_diffusion=False):
        
        utils.print_color(f'ddim steps: {self.ddim_num_inference_steps}', c='y')
        
        device = self.betas.device
        batch_size = shape[0]

        x = self.var_temp * torch.randn(shape, device=device)

        x = apply_conditioning(x, cond, 0) # start from dim 0, different from diffuser

        if return_diffusion: diffusion = [x]
        # 100 // 20 = 5
        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        walls_loc = walls_loc.to(device)

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in time_idx: # if np array, i is <class 'numpy.int64'>
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            assert not walls_loc.requires_grad

            x = self.ddim_p_sample(x, cond, timesteps, walls_loc, eta=0.0, use_clipped_model_output=True)

            x = apply_conditioning(x, cond, 0)

            # progress.update({'t': i})

            x = x.detach()

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x





    def pred_x_tm1_ddim(self, x, t: torch.Tensor, comp_eps, eta, clip_denoised=True):
        '''
        get the sample x_{t-1} when using ddim
        '''
        ## check if ddim steps aligns
        prev_timestep = t - self.num_train_timesteps // self.ddim_num_inference_steps 
        
        print(f'comp ddim, eta: {eta}; t: {t[0]}. prev t: {prev_timestep[0]}, {t.shape}')
        alpha_prod_t = extract(self.alphas_cumprod, t, x.shape) # 
        if prev_timestep[0] >= 0:
            alpha_prod_t_prev = extract(self.alphas_cumprod, prev_timestep, x.shape) # tensor 
        else:
            # extract from a tensor of size 1, cuda tensor [80, 1, 1]
            alpha_prod_t_prev = extract(self.final_alpha_cumprod.to(t.device), torch.zeros_like(t), x.shape)
            print(f'comp ddim alpha_prod_t_prev {alpha_prod_t_prev.shape} {alpha_prod_t_prev[0].item()}')
        assert alpha_prod_t.shape == alpha_prod_t_prev.shape

        beta_prod_t = 1 - alpha_prod_t
        model_output = comp_eps
        
        x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * comp_eps

        x_recon.clamp_(-1., 1.)


        ## 5. compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) \
            * ( 1 - alpha_prod_t / alpha_prod_t_prev )

        std_dev_t = eta * variance ** (0.5)
        
        use_clipped_model_output = True
        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            sample = x
            pred_original_sample = x_recon
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            variance_noise = torch.randn_like(prev_sample)
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance
            

        
        return prev_sample

