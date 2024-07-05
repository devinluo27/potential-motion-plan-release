import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        """dim should be a even number"""
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x: should be (B, 1) -> (B, 1, dim)
        or
        x: should be (B,) -> (B, dim)

        [:,None] is use to unsequence
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Conv1dBlock_dd(nn.Module):
    '''
        Conv1d --> GroupNorm (8 /32) --> (Mish / SiLU)
        ## checkparam groupnorm, n_groups
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8, conv_zero_init=False):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            zero_module( nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2), conv_zero_init ),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)



# --------------------------------------------------------
# -------------- from reduce reuse recycle --------------


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.swish = swish

    def forward(self, x):
        y = super().forward(x.float()).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y

def zero_module(module, do_zero):
    """
    Used for energy parameterization
    Zero out the parameters of a module and return it.
    """
    if do_zero:
        for p in module.parameters():
            p.data.fill_(0)

    return module


def normalization(channels, swish=0.0, num_groups=32):
    """
    Make a standard normalization layer, with an optional swish activation.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=num_groups, swish=swish)

# --------------------------------------------------------



#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    ## torch.tensor [a_t0, a_t1, ..., a_t]
    out = a.gather(-1, t)
    ## expand tuple:(1,) * 3 = (1,1,1)
    ## out: (b,1,...)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    ## 0: (B, 4); horizon-1: (B,4)
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x

def set_loss_noise(noise, conditions, action_dim, value=0.0):
    ## 0: (B, 4); horizon-1: (B,4)
    for t in conditions.keys():
        noise[:, t, action_dim:] = value
    return noise

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'state_l2': WeightedStateL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}



class PositionalEncoding2D(nn.Module):
    """
    Copy From SRT, their default: num_octaves=8, start_octave=0
    To positional encode the wall locations
    e.g., dim after encoded: [6=(wall*2) x (num_octaves*2)] = 48  
    """
    def __init__(self, num_octaves=4, start_octave=0, norm_factor=30):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave
        self.norm_factor = norm_factor

    
    def forward(self, coords, rays=None):
        ## must be True because sinPos is for every location
        ## rearrange is done in vit1d
        if coords.ndim == 3: # True for maze2d
            return self.forward_3Dinput(coords, rays)
        else:
            raise NotImplementedError
    

    ## [Not used] Can be deleted
    def forward_3Dinput(self, coords, rays=None):
        # print('coords', coords.shape) ## B, 6
        # embed_fns = [] # not used
        batch_size, num_points, dim = coords.shape
        coords = coords / self.norm_factor

        ## we assume self.start_octaves=0, self.num_octaves=8 in the example below
        # torch.arange(0, 0+8)
        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)

        # to the device if coords
        octaves = octaves.float().to(coords)
        ## (2 ^ [0., 1., ..., 7.]) * pi
        multipliers = 2**octaves * math.pi
        ## after coords: batch_size, num_points, dim, 1
        coords = coords.unsqueeze(-1)
        ## multipliers: (8,) -> (1,1,1,8)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        ## (batch_size, num_points, dim, 1) * (1,1,1,8)
        scaled_coords = coords * multipliers

        ## (batch_size, num_points, dim, 8) -> (batch_size, num_points, dim * 8)
        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        ## (batch_size, num_points, dim * num_octaves + dim * num_octaves)
        result = torch.cat((sines, cosines), -1)
        return result


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype