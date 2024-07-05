import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pdb
from diffuser.models.helpers import PositionalEncoding2D

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., mish=True):
        super().__init__()
        print(dim, hidden_dim)

        act_fn = nn.GELU() if mish else nn.SiLU()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., attn_proj_out=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = (not (heads == 1 and dim_head == dim)) or attn_proj_out

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., attn_proj_out=False, mish=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, attn_proj_out=attn_proj_out)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, mish=mish))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x




class ViT1D(nn.Module):
    """
    ViT from https://github.com/lucidrains/vit-pytorch#vision-transformer---pytorch
    size: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L621
    original vit: dim -> now embed_dim
    config of a vit-b _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )
    Our default config: embed_dim=32, depth=3, num_heads=1

    *** dim is input and output to the Transformer module,
        the actually attention dim is [num_heads * dim_head].
    """ # img_size, set channels=1
    def __init__(self, *, input_size, patch_size, num_classes, embed_dim, depth, num_heads, mlp_ratio, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., tf_config={}, **kwargs):
        """
        B, input_size, (c)
        input_size: int
        kwargs:
        attn_proj_out (bool): ## explicit control if use a linear project out linear in attention module
        """

        super().__init__()

        # adapter
        self.embed_dim = embed_dim
        dim = embed_dim ## dim of the transformer
        self.patch_size = patch_size
        mlp_dim = embed_dim * mlp_ratio

        heads = num_heads
        print(f'init vanilla ViT1D patch_size:{patch_size}')
        assert input_size % patch_size == 0, 'input dimensions must be divisible by the patch size.'

        self.num_classes = num_classes
        # self.return_features = kwargs.get('return_features', False)
        self.return_cls_token_only = kwargs.get('return_cls_token_only', True)

        num_patches = input_size // patch_size

        ## Our input is [B, XX, channels]
        patch_dim = channels * patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert kwargs['attn_dim'] == num_heads * dim_head
        
        ## Two Spec for input to ViT 
        ## "b_i": (B, input_size); "b_i_c": (B, input_size, channels)
        self.input_spec = tf_config.get('input_spec', 'b_i') 
        self.mish = tf_config.get('mish', True)


        ## e.g. default rearrange to (B, n_walls, 2)
        if self.input_spec == 'b_i': 
            rearrg = Rearrange('b (n_walls psize) -> b n_walls psize', psize = patch_size)
        elif self.input_spec == 'b_i_c':
            rearrg = Rearrange('b (n_walls psize) c -> b n_walls (psize c)', psize = patch_size)
            assert False
        else: 
            raise NotImplementedError()

        

        wall_sinPosEmb = tf_config.get('wall_sinPosEmb', False)
        if wall_sinPosEmb: # a dict
            patch_dim = patch_dim * wall_sinPosEmb['num_octaves'] * 2 ## maze2d:2*4*2=16; kuka:3*4*2=24
            to_patch_embedding = [
                                rearrg,
                                PositionalEncoding2D(num_octaves=wall_sinPosEmb['num_octaves'],
                                                     norm_factor=wall_sinPosEmb['norm_factor']),
                                nn.LayerNorm(patch_dim)]
        else:
            to_patch_embedding = [rearrg,]
        
        to_patch_embedding.extend(
               [nn.Linear(patch_dim, dim),
                nn.Linear(dim, dim * 4),
                nn.Mish() if self.mish else nn.SiLU(),
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim),]
        )
        self.to_patch_embedding = nn.Sequential(*to_patch_embedding)



        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        if tf_config.get('dyn_posemb', None):
            self.dyn_posemb = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        else:
            self.dyn_posemb = None
        if input_size > 50 and patch_size <= 3:
            assert tf_config.get('dyn_posemb', None)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # print(dim, depth, heads, dim_head, mlp_dim, dropout)

        attn_proj_out = kwargs.get('attn_proj_out', False) # not big deal to `or` False
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_proj_out, mish=self.mish)

        self.pool = pool
        self.to_latent = nn.Identity()

        if self.num_classes > 0:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, walls_loc):
        '''
            walls_loc: default [batch, 6]; (can also support [batch, 6, 1])
            output: torch.Size([B, input_size+1, 32])
        '''

        x = self.to_patch_embedding(walls_loc)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.dyn_posemb is not None:
            # pdb.set_trace()
            # B,48+1,dim
            x += self.dyn_posemb # [:, :(n + 1)] # TODO make it optional
        x = self.dropout(x)

        x = self.transformer(x)

        if self.num_classes > 0:
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            return self.mlp_head(x)
        else:
            # if self.return_features:
                # return x[:,1:,:]
            if self.return_cls_token_only:
                return x[:,0,:]
            return x
        



