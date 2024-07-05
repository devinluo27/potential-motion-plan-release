import numpy as np
from PIL import Image
import torch
import random
from colorama import Fore
import pandas as pd

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_dot_dict(**kwargs):
    return DotDict(dict(**kwargs))    

    
def save_gif(imgs, gif_name, duration=50):
    # Setup the 4 dimensional array
    a_frames = []
    for img in imgs:
        a_frames.append(np.asarray(img))
    a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ims[0].save(gif_name, save_all=True, append_images=ims[1:], loop=0, duration=duration)
    print(f'[save_gif] {gif_name}')
    
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def to_np(tensor):
    return tensor.data.cpu().numpy()


def print_color(s, c='r'):
    if c == 'r':
        print(Fore.RED + s + Fore.RESET)
    elif c == 'b':
        print(Fore.BLUE + s + Fore.RESET)
    elif c == 'c':
        print(Fore.CYAN + s + Fore.RESET)
    else:
        print(Fore.YELLOW + s + Fore.RESET)

def print_stat_load(is_suc, pl_time, n_colchk, ):
    pl_time_mean = pl_time.mean( axis=1 )
    n_colchk_mean = n_colchk.mean( axis=1 )
    pl = pd.DataFrame( pl_time_mean ).sem().item()
    ck = pd.DataFrame( n_colchk_mean ).sem().item()
    if is_suc is not None:
        i_s = pd.DataFrame( is_suc.mean( axis=1 ) ).sem().item()
        i_s = i_s * 100
        # print( f'{i_s}' )
    else:
        i_s = -1
    # print( f'{pl:.3f}, {ck:.3f}' )
    return i_s, pl, ck