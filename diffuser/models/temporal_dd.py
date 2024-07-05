import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import Conv1dBlock_dd

class ResidualTemporalBlock_dd(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True,conv_zero_init=False, resblock_config={}, **kwargs):
        '''kwargs: place holder for some useless args e.g. wall_embed_dim'''
        super().__init__()
        force_residual_conv = resblock_config.get('force_residual_conv', False)
        time_mlp_config = resblock_config['time_mlp_config']

        convblock_type = Conv1dBlock_dd

        self.blocks = nn.ModuleList([
            convblock_type(inp_channels, out_channels, kernel_size, mish, conv_zero_init=False), # conv_zero_init, only difference bewteen ori and ori2
            convblock_type(out_channels, out_channels, kernel_size, mish, conv_zero_init=conv_zero_init),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        if time_mlp_config == 2:
            self.time_mlp = nn.Sequential(
                act_fn,
                nn.Linear(embed_dim, out_channels * 2),
                act_fn,
                nn.Linear(out_channels * 2, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif time_mlp_config == 3:
            self.time_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                act_fn,
                nn.Linear(embed_dim * 2, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        else:
            self.time_mlp = nn.Sequential(
                act_fn,
                nn.Linear(embed_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )

        if not force_residual_conv:
            self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
                if inp_channels != out_channels else nn.Identity()
        else:
            self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1)

    def forward(self, x, t, w=None):
        '''
            pipeline:
            1. process x only
            2. process t only
            3. process (x + t) *zero init*
            4. process skip connection

            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            w : placeholder
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)