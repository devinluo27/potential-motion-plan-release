from typing import Union
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os, json

def save_gif(imgs, gif_name, duration=50, caption=''):
    '''this is actually for showing the diffusion process'''
    # Setup the 4 dimensional array
    if type(imgs) == np.ndarray:
        a = imgs
    else:
        a_frames = []
        for img in imgs:
            a_frames.append(np.asarray(img))
        a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ims.insert(0, ims[-1])
    ts = len(ims)
    duration = [duration] * ts
    duration[0] = 2000
    duration[-1] = 8000
    ims[0].save(gif_name, save_all=True, append_images=ims[1:], loop=0, duration=duration)
    print(f'Saved {ts} timesteps to: {gif_name}')


def save_gif_ethucy(imgs, gif_name, duration=50, caption=''):
    '''this is for realworld dataset'''
    # Setup the 4 dimensional array
    if type(imgs) == np.ndarray:
        a = imgs
    else:
        a_frames = []
        for img in imgs:
            a_frames.append(np.asarray(img))
        a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ts = len(ims)
    if type(duration) == int:
        duration = [duration] * ts
        duration[0] = 1000
        duration[-1] = 5000
    else:
        pass
    ims[0].save(gif_name, save_all=True, append_images=ims[1:], loop=0, duration=duration)
    print(f'Saved {ts} timesteps to: {gif_name}')
    

def plot_xy(ys, fname, figtext=''):
    fig, ax = plt.subplots()
    for k,v in ys.items():
        px = np.linspace(1, len(v), len(v))
        ax.plot(px, v, label=k)
    ax.legend()
    time_now = datetime.now().strftime("%m%d-%H%M%S")
    fname = fname.replace('.png', f'{time_now}.png')
    # plt.figtext(0.5, 0.01, figtext, wrap=True, horizontalalignment='center', fontsize=12)
    ax.set_xlabel(figtext)
    fig.savefig(fname=fname)
    

def get_normal_dist(x):
    distribution = torch.distributions.normal.Normal(
        loc=torch.zeros_like(x),
        scale=torch.ones_like(x),
    )
    distribution = torch.distributions.independent.Independent(
        base_distribution=distribution,
        reinterpreted_batch_ndims=2,
    )
    return distribution


def batch_repeat_tensor(x, cond, t, w, n_rp):
    '''
    deepcopy tensor along batch dim for eval pipeline
    '''
    x = x.repeat( (n_rp, 1, 1) )
    cond_2 = {}
    for k in cond:
        cond_2[k] = cond[k].repeat( (n_rp, 1,) ) # 2d (B,2)
    t = t.repeat( (n_rp,) ) # (B,)
    w = w.repeat( (n_rp, 1,) ) # 2d

    return x, cond_2, t, w
    

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def filter_json_serializable(input_dict):
    return {k: v for k, v in input_dict.items() if is_json_serializable(v)}
