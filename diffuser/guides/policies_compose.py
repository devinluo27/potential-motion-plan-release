import torch, pdb
import diffuser.utils as utils
from diffuser.models import GaussianDiffusionPB
from .policies import Policy
from .policies import Trajectories
import numpy as np
from diffuser.models.helpers import apply_conditioning
from typing import List


class PolicyCompose(Policy):

    def __init__(self, diffusion_model_list: List[GaussianDiffusionPB], normalizer_list, use_normed_wallLoc, 
                 use_ddim, po_config,):
        '''
        Args:
            diffusion_model_list: a list of diffusion models (can be the same model or different models)
            normalizer_list: a list of normalizer (will be used to normalize the trajectory of states)
            use_normed_wallLoc: do normalization for wall locations, default False
            use_ddim: if use ddim sampling, default True
            po_config: other configs
        '''
        self.diffusion_model_list = diffusion_model_list
        self.normalizer_list = normalizer_list
        self.action_dim = normalizer_list[0].action_dim

        self.n_models = len(diffusion_model_list)
        assert self.n_models == 2, 'currently support 2 models.'

        self.num_walls_c = torch.tensor( po_config['num_walls_c'] )
        self.wall_dim = po_config['wall_dim']
        self.comp_type = po_config['comp_type'] # 
        # self.rng = np.random.default_rng(27)
        assert self.comp_type in ['direct', 'share']
        self.use_ddim = use_ddim
        self.ddim_steps = po_config['ddim_steps']
        # self.horizon_list = po_config['']
        self.wall_is_dyn = po_config['wall_is_dyn']
        self.is_dyn_env = self.wall_is_dyn is not None
        self.uncond_base_idx: int = po_config['uncond_base_idx']
        self.ddim_eta = po_config['ddim_eta']
        
        self.use_normed_wallLoc = use_normed_wallLoc
        self.w_split_sizes = tuple( (self.num_walls_c * self.wall_dim).to(torch.int32) )
        utils.print_color(f'PoCo num_walls_c: {self.num_walls_c} {self.wall_dim}')

        ## use the first model to get info; keep the name for consistency with parent class
        self.diffusion_model = diffusion_model_list[0]
        
        self.diffusion_model.eval()
        for dm in diffusion_model_list:
            assert dm.horizon == self.diffusion_model.horizon
            dm.eval()
        self.is_same_model = True if self.diffusion_model_list[0] == self.diffusion_model_list[1] else False

        self.normalizer = normalizer_list[0]
        ## num of timestep of diffusion model
        self.n_timesteps = self.diffusion_model.n_timesteps
        self.horizon = self.diffusion_model.horizon
        self.transition_dim = self.diffusion_model.transition_dim
        self.observation_dim = self.diffusion_model.observation_dim
        self.sampler = None
        self.use_mala_sampler = False


        for n in normalizer_list:
            assert n.action_dim == normalizer_list[0].action_dim
            assert n.observation_dim == normalizer_list[0].observation_dim
            # normalizer check
            assert (n.normalizers['observations'].mins == \
                        normalizer_list[0].normalizers['observations'].mins).all()
            assert (n.normalizers['observations'].maxs == \
                        normalizer_list[0].normalizers['observations'].maxs).all()
        
        ## cls-free guidance weight for each model
        cg_w = torch.tensor([ dm.condition_guidance_w for dm in self.diffusion_model_list ])
        self.cg_w = cg_w.reshape(self.n_models, 1, 1, 1).to('cuda')
        ## if store the x0 of every denosiing steps
        self.return_diffusion = False


    def _format_conditions(self, conditions, batch_size, norm_idx):
        '''norm_idx: use which normalize'''
        conditions = utils.apply_dict(
            self.normalizer_list[norm_idx].normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        if batch_size > 0:
            raise RuntimeError
            
        return conditions
    
    def wallLoc_preproc(self, wall_c):
        '''
        This function transforms wall_c to the model input format,
        from a list of np with shape [num_walls, dim]  to  a list of tensor with shape [B, num_walls*dim],
        args:
            w (3d np or list of 2d np)
        '''

        ## 1. direct: directly compose two models, i.e., no wall locations overlap
        if self.comp_type == 'direct':
            
            if self.is_dyn_env:
                n_w = round(self.num_walls_c.sum().item())
                # wall_c: B,h*nw*d  ->  B, h, nw, d 
                wall_c = utils.dyn_wallLoc_b_h_nw_d(wall_c, n_w, self.wall_dim)
                # wall_c: B,h*nw*d  ->  B, h, nw*d
                wall_c = torch.flatten( wall_c, start_dim=2, end_dim=3)
                # tuple ( (B, h, nw1*d), (B, h, nw2*d) )
                wall_c = torch.split(wall_c, self.w_split_sizes, dim=2)
                
                if np.array(self.wall_is_dyn, dtype=bool).all():
                    raise NotImplementedError() # composed two dynamic
                else:
                    # first model dynamic, second model static
                    # wall_c[1].shape [1, 48, 6] -> [1, 6]
                    if self.wall_is_dyn[0]:
                        tmp = [ torch.flatten(wall_c[0], 1, 2), wall_c[1][:, 0, :] ]
                    else:
                        tmp = [ wall_c[0][:, 0, :], torch.flatten(wall_c[1], 1, 2), ]
                    wall_c = tmp
                

            else:
                ## static, B, nw*dim -> ( (B, nw1*d), (B, nw2*d) )
                ## split to two parts, no overlap
                wall_c = torch.split(wall_c, self.w_split_sizes, dim=1)

                if self.w_split_sizes[1] == 2:
                    wall_c = list(wall_c)
                    w_tmp = torch.repeat_interleave(wall_c[1], repeats=2, dim=1) # B, 2 -> B, 4
                    w_tmp = w_tmp + torch.randn_like(w_tmp) * 0.03 # add noise
                    wall_c[1] = torch.cat( [wall_c[1], w_tmp], dim=1 ) # B, 6
                elif self.w_split_sizes[1] == 4:
                    wall_c = list(wall_c)
                    w_tmp = torch.clone(wall_c[1][:, :2])
                    w_tmp = w_tmp + torch.randn_like(w_tmp) * 0.03 # add noise
                    wall_c[1] = torch.cat( [wall_c[1], w_tmp], dim=1 )
                    
                    


        ## 2. share, with wall locations overlappings
        # 7 walls -> 6 + 6
        elif self.comp_type in ['share', 'share-model2fix']:
            from diffuser.datasets.preprocessing import shuffle_along_axis # no need
            assert self.n_models == 2
            B, nwd = wall_c.shape
            nw = round(nwd / self.wall_dim) # int
            wall_c = wall_c.reshape(B, -1, self.wall_dim)
            
            ## tmp1: wall locations of the first part
            ## tmp2: wall locations of the second part
            tmp1 = torch.clone( wall_c[:, :self.num_walls_c[0], :] ).flatten(1, 2)
            if self.comp_type == 'share':
                ## this is used in generalization to increasing obstalces
                ## e.g., we have model trained on 6 obstacles and now generalizing to 8 obstacles
                ## nw=8, num_walls_c=[6,2] -> tmp2_idx=2
                ## So wall idxs 1: [0,1,2,3,4,5,]; wall idxs 2: [2,3,4,5,6,7]
                tmp2_idx = (nw - self.num_walls_c[0]) #
            elif self.comp_type == 'share-model2fix':
                ## this is used in generalization to different static obstalces
                ## nw=9, num_walls_c=[6,3] -> tmp2_idx=2
                ## So wall idxs 1: [0,1,2,3,4,5,]; wall idxs 2: [6,7,8]
                # which makes sure that the later model only take given number of obstacles
                tmp2_idx = max(0, (nw - self.num_walls_c[1]) )

            tmp2 = torch.clone( wall_c[:, tmp2_idx:, :] ).flatten(1, 2)

            if self.comp_type == 'share':
                assert tmp1.shape == tmp2.shape

            wall_c = [tmp1, tmp2]

        else:
            raise NotImplementedError()

        return wall_c
    

    def call_multi_horizon(self, cond_mulho: List, wall_c):
        '''
        generates motion plans for each horizons in a forloop, 
        return a list of results
        '''
        self.multi_hor_x0_perstep = []
        samples_mulho = []
        for cond in cond_mulho:

            if torch.is_tensor(wall_c):
                wall_c = wall_c.detach()
            _, samples = self.__call__(cond, wall_c)
            samples_mulho.append( samples )
            self.multi_hor_x0_perstep.append(getattr(self, 'x0_perstep', None))

        return samples_mulho
    


    

    def __call__(self, cond, wall_c: torch.Tensor):
        '''
        Args:
            cond: a dict of tensor, *not normalized*, in cpu
            wall_c: positions of all presented walls 
        '''
        assert len(cond[0].shape) == 2

        assert torch.is_tensor( cond[0] ) and cond[0].device == torch.device('cpu')
        ## here, We normalize the cond so that it can be applied directly,
        ## new dict is returned each time, no data overlap, no deepcopy needed here
        ## cond 0:start, h-1:goal
        cond_list = [self._format_conditions(cond, -1, n_idx) for n_idx in range(self.n_models)]
        batch_size = cond_list[0][0].shape[0]

        
        ## cls-free guidance weight for each model
        cg_w = torch.tensor([ dm.condition_guidance_w for dm in self.diffusion_model_list ])
        cg_w = cg_w.reshape(self.n_models, 1, 1, 1).to('cuda')
        
        # tuple ( [B, nw1*2], [B, nw2*2] )
        wall_c = self.wallLoc_preproc(wall_c)

        ## 1. prepare latent
        if type(self.diffusion_model) == GaussianDiffusionPB:
            dim0 = self.observation_dim
        else:
            dim0 = self.observation_dim
            utils.print_color( f'Compose {type(self.diffusion_model)}' )


        ## potentially [Caution], infer from cond 
        horizon = sorted( list( cond_list[0].keys() ) )[-1] + 1 # [0, 47]
        assert horizon >= 20, 'prevent bugs'
        shape = (batch_size, horizon, dim0)

        x = self.diffusion_model.var_temp * torch.randn(shape, device=self.device)

        # x = apply_conditioning(x, cond, dim1)
        ## B, horizon, dim
        latents = x
        
        if self.use_ddim:
            latents = self.ddim_sample_loop(latents, cond_list, wall_c, batch_size)
        else:
            latents = self.ddpm_sample_loop(latents, cond_list, wall_c, batch_size)

        ## cond, latents are in cuda
        latents = apply_conditioning(latents, cond_list[0], 0)


        ## now latents is the output
        _, trajectories = self.output_traj_postproc(latents)
        
        if self.return_diffusion:
            for i_ps in range(len(self.x0_perstep)):
                apply_conditioning(self.x0_perstep[i_ps], cond_list[0], 0)
                ## we only needs the second one ([1]), which is trajectory
                self.x0_perstep[i_ps] = self.output_traj_postproc(self.x0_perstep[i_ps])[1]



        return _, trajectories
    

    def ddpm_sample_loop(self, x_T, cond_list, wall_c, batch_size):
        verbose = True
        latents = x_T # highest level noise
        cg_w = torch.clone(self.cg_w)
        ## warmup step?
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        reverse_timesteps = np.arange(self.n_timesteps-1, -1, -1)
        for i, t in enumerate(reverse_timesteps):
            # print('batch_size', type(batch_size), 't', type(t))
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            ## why (cond uncond in one batch) - (separate batch) is not zero ?? residual level 1e-5
            ## this impl is about 0.5 * less time, do not need energy here
            (_, eps_conds), (_, eps_unconds) = self.models_pred(latents, cond_list, timesteps, wall_c)
            
            ## --- do classifer free guidance ---
            diff = eps_conds - eps_unconds

            ## [n_models, 1, 1, 1] * [n_models, bsize, horizon, dim]
            w_diff = cg_w * diff

            eps_unconds = eps_unconds[self.uncond_base_idx]
            ## [bsize, horizon, dim]
            composed_epsilon = eps_unconds + w_diff.sum(dim=0, keepdims=False)

            ## [NOTE] 1.4 compute the previous noisy sample x_t -> x_t-1
            # print('latents 1', latents.shape, 'composed_epsilon', composed_epsilon.shape)
            latents = self.diffusion_model_list[0].pred_x_tm1_luo(latents, timesteps, composed_epsilon, clip_denoised=True)
            # progress.update({'t': t})

        progress.close()
        x_0 = latents # output x_0
        return x_0
    

    def ddim_sample_loop(self, x_T, cond_list, wall_c, batch_size):
        verbose = True
        latents = x_T # highest level noise
        cg_w = torch.clone(self.cg_w)
        ## warmup step?
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        
        # reverse_timesteps = np.arange(self.n_timesteps-1, -1, -1)

        time_idx = self.diffusion_model_list[0].ddim_set_timesteps(self.ddim_steps)
        utils.print_color(f'ddim steps: {self.ddim_steps}')
        self.x0_perstep = []

        for i, t in enumerate(time_idx):
            # print('batch_size', type(batch_size), 't', type(t))
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            ## stack the cond and uncond in one batch
            ## this impl is about 0.5 * less time
            (_, eps_conds), (_, eps_unconds) = self.models_pred(latents, cond_list, timesteps, wall_c)
            
            ## --- do classifer free guidance ---
            diff = eps_conds - eps_unconds

            ## [n_models, 1, 1, 1] * [n_models, bsize, horizon, dim]
            w_diff = cg_w * diff

            ## NOTE: we can use the first as uncond base or specified by self.uncond_base_idx
            # eps_unconds = eps_unconds[0]
            eps_unconds = eps_unconds[self.uncond_base_idx]
            ## [bsize, horizon, dim]
            composed_epsilon = eps_unconds + w_diff.sum(dim=0, keepdims=False)


            ## latents = composed_epsilon = [1, 144, 6]
            ## [NOTE] compute the previous noisy sample x_t -> x_t-1
            # print('latents 1', latents.shape, 'composed_epsilon', composed_epsilon.shape)
            latents = self.diffusion_model_list[0].pred_x_tm1_ddim(latents, timesteps, composed_epsilon, eta=self.ddim_eta, clip_denoised=True)
            # progress.update({'t': t})
            if self.return_diffusion:
                self.x0_perstep.append(latents.detach())


        progress.close()
        x_0 = latents # output x_0
        return x_0


    def models_pred(self, x_t, cond_list: list, t, wall_c: list):
        '''
        cond_list: list of dict
        t: tensor (B,)
        wall_c: list of tensor
        '''
        eps_conds = []
        eps_unconds = []
        engy_conds, engy_unconds = [], []
        for i_m in range(self.n_models):
            x_t = x_t.detach()
            diffusion_model = self.diffusion_model_list[i_m]
            assert not diffusion_model.training
            ## run reverse diffusion process
            
            # assert type(diffusion_model) == GaussianInvDynDiffusion

            wall_locations = wall_c[i_m]
            assert wall_locations is not None
            if self.use_normed_wallLoc:
                wall_locations = self.normalizer_list[i_m].normalize(wall_locations, 'infos/wall_locations')
            
            cond = cond_list[i_m]
            # tp: tuple
            cond_tp, uncond_tp = diffusion_model.pred_unet(x_t, cond, t, wall_locations,
                                                                     return_epsilon=True,  mala_sampler=self.use_mala_sampler)
            engy_conds.append(cond_tp[0])
            engy_unconds.append( uncond_tp[0])

            eps_conds.append( cond_tp[1] )
            eps_unconds.append( uncond_tp[1] )
            

        engy_conds = torch.stack( engy_conds, dim=0 ) if self.use_mala_sampler else None
        engy_unconds = torch.stack( engy_unconds, dim=0 ) if self.use_mala_sampler else None
        eps_conds = torch.stack( eps_conds, dim=0 ) # (2, B, h, dim)
        eps_unconds = torch.stack( eps_unconds, dim=0 )

        return (engy_conds, eps_conds), (engy_unconds, eps_unconds) # from here 19:15
        




  

    def ddim_replan(self, traj, cond, wall_c, dn_steps):
        '''
        Note that replan only accept a cond
        traj (tensor): (1, h, d) a clean traj; every h is ok?
        cond (dict): normed: 0: tensor(dim,), h-1: dim,
        batch_size: 1
        '''
        # cond_list = [self._format_conditions(cond, -1, n_idx) for n_idx in range(self.n_models)]
        cond_list = [cond for _ in range(self.n_models)] # shallow copy

        cg_w = torch.clone(self.cg_w)
        # tuple ( [B, nw1*2], [B, nw2*2] )
        wall_c = self.wallLoc_preproc(wall_c)
        
        time_idx = self.diffusion_model_list[0].ddim_set_timesteps(self.ddim_steps)
        utils.print_color(f'ddim replan steps: {dn_steps}')


        # 1. -- get noisy traj --
        # 1, h, dim; noise level difined by dn_steps
        noisy_traj = self.diffusion_model_list[0].ddim_get_noisy_traj(traj, self.ddim_steps, dn_steps)
        
        x = noisy_traj

        for i, t in enumerate(time_idx[-dn_steps:]):
            timesteps = torch.full((1,), t, device=self.device, dtype=torch.long)
            
            (_, eps_conds), (_, eps_unconds) = self.models_pred(x, cond_list, timesteps, wall_c)
            
            diff = eps_conds - eps_unconds
            w_diff = cg_w * diff

            eps_unconds = eps_unconds[self.uncond_base_idx]
            ## [bsize, horizon, dim]
            composed_epsilon = eps_unconds + w_diff.sum(dim=0, keepdims=False)
            
            ## [NOTE] 1.4 compute the previous noisy sample x_t -> x_t-1
            x = self.diffusion_model_list[0].pred_x_tm1_ddim(x, timesteps, composed_epsilon, eta=self.ddim_eta, clip_denoised=True)
            x = x.detach()

        if traj.ndim == 2: # return h, dim
            x = x[0]

        return x
    
    def set_ddim_steps(self, ddim_steps):
        self.ddim_steps = ddim_steps
        ## set the # of step for each model
        for d_model in self.diffusion_model_list:
            d_model.ddim_num_inference_steps = ddim_steps
        






    

    

    