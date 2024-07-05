import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
import diffuser.utils as utils

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    PositionalEncoding2D,
    Conv1dBlock_dd,
)
from .vit_vanilla import ViT1D
import numpy as np
from .temporal_dd import ResidualTemporalBlock_dd


class ResidualTemporalBlock_WCond(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, wall_embed_dim, 
                 kernel_size=5, **kwargs):
        """
        embed_dim: input dimension for time_mlp (dimension of time positional embedding)
        wall_embed_dim:
        """
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.wallLoc_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(wall_embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, walls_loc):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''

        out = self.blocks[0](x) + self.time_mlp(t) + self.wallLoc_mlp(walls_loc)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

# --------------------------------------------------------

## WCond: Condtioned on Walls
class TemporalUnet_WCond(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32, # may use 64
        dim_mults=(1, 2, 4, 8),
        num_walls=3,
        wall_embed_dim=32,
        wall_sinPosEmb={},
        network_config={},
    ):
        """
        NOTE energy_mode might be implemented only when cat_t_w=True,
        e.g. some zero init is not impled in residual block.
        wall_embed_dim: embed dim of all walls

        """
        super().__init__()

        ## dim=64 [2,64*1,64*4,64*8]
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        ## [(64,128), (128,256), (256,512)]
        in_out = list(zip(dims[:-1], dims[1:]))
        utils.print_color(f'[ models/temporal_cond ] Channel dimensions: {in_out}', c='c')

        ## --------- init MLP for time / wall ---------
        ## cat the vector embedding of time and wall before feeding to the MLP
        self.cat_t_w = network_config.get('cat_t_w', False)
        self.resblock_ksize = network_config.get('resblock_ksize', 5) # kernel size for residual block
        self.use_downup_sample = network_config.get('use_downup_sample', True)

        assert self.use_downup_sample and self.resblock_ksize == 5, 'the default settings'
        
        if self.cat_t_w:
            time_dim = dim + wall_embed_dim
        else:
            time_dim = dim

        ## set param used in ebm
        self.energy_mode = network_config.get('energy_mode', False)
        if self.energy_mode:
            mish = False
            act_fn = nn.SiLU()

            self.energy_param_type = network_config['energy_param_type'] ## should use this line
            if 'L2' in self.energy_param_type:
                self.conv_zero_init = network_config.get('conv_zero_init', False)
               
            else: 
                raise NotImplementedError()
            
            print(f'[ models/temporal_cond ] conv_zero_init {self.energy_param_type} {self.conv_zero_init}')
        else:
            mish = True
            act_fn = nn.Mish()
            self.conv_zero_init = False


        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.network_config = network_config

        self.wallLoc_encoder_type = network_config.get('wallLoc_encoder', 'mlp').lower()
        ## default no dropout
        self.concept_drop_prob = network_config.get('concept_drop_prob', -1.0)
        self.last_conv_ksize = network_config.get('last_conv_ksize', 1) # 1 is more stable than 5
        self.force_residual_conv = network_config.get('force_residual_conv', False)
        self.time_mlp_config = network_config.get('time_mlp_config', False)
        resblock_config = dict(force_residual_conv=self.force_residual_conv,
                               time_mlp_config=self.time_mlp_config)
        
        assert not self.force_residual_conv, 'must be False'
        assert self.last_conv_ksize == 1, '1 is from diffuser'


        print(f'[TemporalUnet_WCond] concept_drop_prob: {self.concept_drop_prob}')
        print(f'[TemporalUnet_WCond] time_dim: {time_dim}')
        
        # --------- init encoder for wall locations ---------
        if self.wallLoc_encoder_type == 'mlp':
            '''construct mlp encoder by default'''
            if wall_sinPosEmb:
                ## use pos embbeding
                num_octaves = wall_sinPosEmb['num_octaves']
                walls_input_dim = num_octaves * 2 * (num_walls * 2)
                wallLoc_mlp_list = [PositionalEncoding2D(num_octaves=num_octaves,),]
            else:
                ## no sin pos embedding
                walls_input_dim = num_walls * network_config.get('wall_dim', 2)
                wallLoc_mlp_list = []

            wallLoc_mlp_list.extend([
                nn.Linear(walls_input_dim, wall_embed_dim),
                nn.Linear(wall_embed_dim, wall_embed_dim * 4),
                act_fn,
                nn.Linear(wall_embed_dim * 4, wall_embed_dim),
            ])
            self.wallLoc_encoder = nn.Sequential(*wallLoc_mlp_list)
        elif self.wallLoc_encoder_type == 'vit1d':
            self.vit1d_config = network_config['vit1d_config']
            print('[TemporalUnet_WCond] construct ViT1D as wall loc encoder')
            print(f'self.vit1d_config: {self.vit1d_config}')
            self.wallLoc_encoder = ViT1D(**self.vit1d_config)
        
        else: 
            raise NotImplementedError()


        print('[TemporalUnet_WCond] self.wallLoc_encoder', type(self.wallLoc_encoder))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        ## num_resolutions is the number of layer in UNet?
        print('[TemporalUnet_WCond]: in_out: ', in_out, f'num_walls:{num_walls}')

        res_block_type = ResidualTemporalBlock_dd if self.cat_t_w else ResidualTemporalBlock_WCond

        

        self.down_times = network_config.get('down_times', 1e5)
        utils.print_color(f'[Unet down_times] {self.down_times}', c='c')
        ## default in_out: [(64,128), (128,256), (256,512)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind >= self.down_times

            self.downs.append(nn.ModuleList([
                res_block_type(dim_in, dim_out, embed_dim=time_dim, horizon=horizon,wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init,resblock_config=resblock_config, kernel_size=self.resblock_ksize), # ks should be 5 by default
                res_block_type(dim_out, dim_out, embed_dim=time_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init,resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                Downsample1d(dim_out) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize)
        self.mid_block2 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind < ( num_resolutions - self.down_times - 1)

            ##? Eg. dim_out:4, dim_in:8, dim_out*2 because we concat residual 
            self.ups.append(nn.ModuleList([
                res_block_type(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                res_block_type(dim_in, dim_in, embed_dim=time_dim, horizon=horizon, wall_embed_dim=wall_embed_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                Upsample1d(dim_in) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2
        
        ## -- Ordinary Diffusion Setup --
        if not self.energy_mode:
            self.final_conv = nn.Sequential(
                Conv1dBlock(dim, dim, kernel_size=self.resblock_ksize), # 5
                nn.Conv1d(dim, transition_dim, 1),
            )
        ## -- Energy Diffusion Parameterization Setup --
        elif self.energy_param_type == 'L2':
            self.final_conv = nn.Sequential(
                Conv1dBlock_dd(dim, dim, kernel_size=5, mish=mish, conv_zero_init=False),
                nn.Conv1d(dim, transition_dim, 1),
            )
        else:
            raise NotImplementedError()



    def forward(self, x, cond, time, walls_loc, use_dropout=True, 
                force_dropout=False, half_fd=False,):
        '''
            x : [ batch x horizon x transition ]
            time: [batch,]
            walls_loc: [batch, 6], 2D
            half_fd: drop the conditions for the second half in the input batch 
        '''
        if self.energy_mode:
            x.requires_grad_(True)
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        ## Encode Wall Locations to a feature vector w
        w = self.wallLoc_encoder(walls_loc)
        ## what we want is like [B, dim], use the cls_token if vit1d

        ## drop concept only when training, rand uniform [0, 1)
        if use_dropout:
            assert self.training
            b = w.shape[0]
            w[np.random.rand(b,) < self.concept_drop_prob] = 0.


        if force_dropout:
            assert not self.training
            if half_fd:
                # drop the second half
                assert len(w) % 2 == 0
                w[int(len(w)//2):] = 0. * w[int(len(w)//2):] 
            else:
                w = 0. * w

        if self.cat_t_w:
            t = torch.cat([t,w], dim=-1)
        
        h = []

        for resnet, resnet2, downsample in self.downs:

            x = resnet(x, t, w)
            x = resnet2(x, t, w)
            h.append(x)
            x = downsample(x)

        # print(f'after downs: {x.shape}')

        x = self.mid_block1(x, t, w)
        x = self.mid_block2(x, t, w)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, w)
            x = resnet2(x, t, w)
            x = upsample(x)

        # print(f'after ups: {x.shape}')

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        ## energy_mode will return inside
        if self.energy_mode:
            unet_out = x # B, horizon, dim
            if self.energy_param_type in ['L2',]:
                energy_norm = 0.5 * unet_out * unet_out # should not have neg sign
                energy_norm = energy_norm.sum(dim=(1,2))
            else: 
                raise NotImplementedError()

            
            if not self.training:
                eps = torch.autograd.grad([energy_norm.sum()],[x_inp],create_graph=False)[0]
                # print('energy_norm.sum()', energy_norm.sum())
            else:
                engy_batch = energy_norm.sum()
                eps = torch.autograd.grad([engy_batch,],[x_inp],create_graph=True)[0]
                return eps, engy_batch.detach()

                 
            return eps

        ## final output: B H dim
        # print(f'final output: {x.shape}')

        return x
