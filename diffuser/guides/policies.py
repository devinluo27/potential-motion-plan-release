from collections import namedtuple
import numpy as np
import torch
import einops
import pdb
import diffuser.utils as utils
from diffuser.models import GaussianDiffusionPB
from typing import List
Trajectories = namedtuple('Trajectories', 'actions observations')

class Policy:
    '''
    A class that wraps a potential diffusion model to generate motion plans
    '''
    def __init__(self, diffusion_model, normalizer, use_ddim=False):
        self.diffusion_model: GaussianDiffusionPB = diffusion_model

        self.diffusion_model.eval()
        self.use_ddim = use_ddim

        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        if batch_size > 0:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                'd -> repeat d', repeat=batch_size,
            )
        return conditions
    

    def call_multi_horizon(self, cond_mulho: List, wall_locations, use_normed_wallLoc):
        '''
        generates motion plans for each horizons in a forloop, 
        return a list of results
        '''
        samples_mulho = []
        for cond in cond_mulho:
            # pdb.set_trace()
            if torch.is_tensor(wall_locations):
                wall_locations = wall_locations.detach()
            _, samples = self.__call__(cond, batch_size=-1, 
                                       wall_locations=wall_locations, use_normed_wallLoc=use_normed_wallLoc,)
            samples_mulho.append( samples )
        return samples_mulho


    def __call__(self, conditions, debug=False, batch_size=1, wall_locations=None, use_normed_wallLoc=False, return_diffusion=False):
        """
        input wall_locations: tensor [batch_size, num_walls*2]
        [NOTE] batch_size: 
        > 0: it is acutally the repeated time of the conditions: shape(dim,).
        < 0: automatically infer bsize, conditions: shape(bsize, dim,).
        *Currently*, the condition of the batch should be the same

        return a tuple of size 2: 
        1.[NOTE] first_action of batch id0 (2,)
        2.the whole trajectory[actions,observations]
        """

        assert not self.diffusion_model.training

        conditions = self._format_conditions(conditions, batch_size)




        horizon = sorted( list( conditions.keys() ) )[-1] + 1 # [0, 47]
        self.diffusion_model.horizon = horizon


        ## run reverse diffusion process
        if type(self.diffusion_model) == GaussianDiffusionPB:
            assert wall_locations is not None
            if use_normed_wallLoc:
                # print('wall_locations', wall_locations) ## checked normed
                assert conditions[0][0].shape[-1] == 2, 'not kuka, not using norm'
                wall_locations = self.normalizer.normalize(wall_locations, 'infos/wall_locations')
                # print('normed wall_locations', wall_locations) ## checked normed

            sample = self.diffusion_model(conditions, wall_locations, return_diffusion=return_diffusion, use_ddim=self.use_ddim)
            if return_diffusion:
                sample, diffusion = sample
        else: 
            raise NotImplementedError(f'type: {type(self.diffusion_model)}')

        action, trajectories = self.output_traj_postproc(sample)

        if return_diffusion:
            dfu_action = []; dfu_obs = []
            for i in range(diffusion.shape[1]):
                ac, traj = self.output_traj_postproc(diffusion[:,i,...])
                dfu_obs.append(traj.observations)
            
            # dfu_action = np.stack(dfu_action, axis=1)
            dfu_obs = np.stack(dfu_obs, axis=1)
            ## diffusion: torch.Size([160, 101, 144, 2])
            ## dfu_action: useless, ignore
            ## dfu_obs (B, n_denoising_steps, Horizon, dim)
            return action, trajectories, dfu_action, dfu_obs
        else:
            return action, trajectories





    def output_traj_postproc(self, sample):
        '''
        sample is the direct output of the model,
        call inv_model inside if any,
        the output is a name tuple Trajectories, traj:(actions, obs)
        '''

        if isinstance(self.diffusion_model, GaussianDiffusionPB):
            ## TODO compute inverse dynamic
            # actions = np.zeros_like(sample[:, :, :self.action_dim])
            # samples is the traj
            bsize, horizon = sample.shape[0:2] 
             
            
            actions = np.zeros(shape=(bsize, horizon, self.diffusion_model.action_dim))
            
        else:
            raise NotImplementedError()
            ## extract action [ batch_size x horizon x transition_dim ]
            actions = sample[:, :, :self.action_dim]
        

        sample = utils.to_np(sample)
        


        ## extract first action, useless
        action = actions[0, 0]


        if isinstance(self.diffusion_model, GaussianDiffusionPB):
            normed_observations = sample
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
        else:
            raise NotImplementedError()
            normed_observations = sample[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations')

        

        trajectories = Trajectories(actions, observations)
        return action, trajectories