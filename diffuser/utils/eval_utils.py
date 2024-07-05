import numpy as np
import torch
import torch.nn.functional as F
import diffuser.utils as utils
from colorama import Fore
import einops, imageio
from datetime import datetime

WALL = 10
EMPTY = 11
GOAL = 12

def maze2d_is_valid_traj(traj, wall_locations_inner):
    """
    normalized or unnormalized ??

    traj Tensor: B, horizon, 2 (dim)
    wall_locations_inner: a list or numpy (num_walls, 2)
    return True iff the path didn't go through a wall
    bool: B,
    """
    assert traj.shape[2] == 2, 'only position is needed'

    # 1. linearly add 10 points between two points in traj
    # traj: (B, horizon, dim)
    traj = interpolate_traj(traj, n_added_points=3)
    

    ## (num_walls, 2)
    # wallLoc_arr = np.array(list(zip(*np.where(maze_arr == WALL))))
    ## (1, num_walls, 2)
    # wallLoc_arr = torch.tensor(wallLoc_arr).unsqueeze(0)


    # 2. prepare wall location as tensor
    if not torch.is_tensor(wall_locations_inner):
        wall_locations_inner = torch.tensor(wall_locations_inner)
    if wall_locations_inner.dim() == 2: 
        wall_locations_inner.unsqueeze_(0)
    
    # 3. check if any points transpass the walls
    bbox = generate_bbox(wall_locations_inner)
    ## maybe with a loose threshold: shrink bbox to some extent 0.1?; (is applied to every dim)
    is_cross = is_traj_cross_bbox(traj, bbox, bbox_shrink=0.1) # (B,) # prev 0.02
    is_valid = ~ is_cross

    return is_valid # shape (B,)

def interpolate_traj(traj, n_added_points=10):
    """ 
    checked
    traj: torch tensor (B, horizon, dim) -> (B, size_interp, dim)
    we only care about the trajectory, not the action.
    n_added_points: num points added between two samples
    """
    assert traj.dim() == 3
    horizon = traj.shape[1]
    size_interp = (horizon - 1) * n_added_points + horizon
    ## (B, dim, num_points)
    traj = traj.transpose(1, 2)
    traj = F.interpolate(traj, size=size_interp, mode='linear', align_corners=True)
    ## (B, size_interp, dim)
    traj = traj.transpose(1, 2)
    return traj



def generate_bbox(location, height=1.,width=1.):
    """
    Generate a bounding box centered at a given location.
    For convert a wall location to a bounding box.
    Args:
        location (torch.Tensor): Coordinates of the center (B,num_walls,2) [x, y].
        width (float): Width of the bounding box.
        height (float): Height of the bounding box.
    Returns:
        tuple (of torch.Tensor): (x_min, y_min, x_max, y_max) shape: (B,num_walls,1)
    """
    assert location.dim() == 3 and location.shape[2] == 2
    ## (B,num_walls,1)
    x, y = location[:, :, :1], location[:, :, 1:2]
    half_width = width / 2
    half_height = height / 2

    x_min = x - half_width
    y_min = y - half_height
    x_max = x + half_width
    y_max = y + half_height
    ## [B, 3, 1]
    # print('y_max', y_max.shape)

    return (x_min, y_min, x_max, y_max)



def is_traj_cross_bbox(traj, box, bbox_shrink=0.0):
    """
    Check if points are inside a bounding box.
    If one point of a trajectory is inside the a wall, then return True.

    Args:
        traj (torch.Tensor): 
        Batch of point coordinates with shape (batch_size, horizon, 2) [x, y].
        box (tuple of torch.Tensor):
        (x_min, y_min, x_max, y_max) shape [B,num_walls,1]

    Returns:
        torch.Tensor: Boolean tensor of shape (batch_size,) indicating whether each point is inside the box.
    """
    ## In the example, 3 is num_walls, 4 is horizon
    ## x:torch.Size([B, 4,]), y:torch.Size([B, 4,])
    ## -->> x:torch.Size([B, 1, 4]), y:torch.Size([B, 1, 4])
    ## x_min: torch.Size([B, 3, 1])
    ## x * x_min: Auto broadcast: [B, 3, 4]

    # print(traj.shape, len(box))
    assert traj.dim() == 3 and len(box) == 4

    x, y = traj[:, :, 0], traj[:, :, 1]
    x, y = traj[:, :, 0], traj[:, :, 1]
    x, y = x.unsqueeze(1), y.unsqueeze(1)
    # print('x\n', x.shape)
    x_min, y_min, x_max, y_max = box
    # print('x_min\n', x_min.shape)

    ## loosen bound
    x_min = x_min + bbox_shrink
    y_min = y_min + bbox_shrink
    x_max = x_max - bbox_shrink
    y_max = y_max - bbox_shrink
    
    ## result shape (B, num_walls, interpolated_horizon)
    result = (x >= x_min) * (x <= x_max) * (y >= y_min) * (y <= y_max)
    # print('result\n', result)
    # print(result.shape)


    ## 1.aggregate along horizon: True if one point cross the wall
    # each points in traj must not cross the wall
    result = torch.any(result, dim=2)
    ## 2. aggregate along walls: should not cross any wall
    result = torch.any(result, dim=1)
    
    # print(result.shape) # [B,]

    return result




def compute_success_rate(total_reward_dict):
    """compute success rate of rollout"""
    num_samples = len(total_reward_dict)
    cnt = 0
    for k,v in total_reward_dict.items():
        if v > 0:
            cnt += 1
    return cnt / num_samples


def compute_avg_score(score_dict):
    num_samples = len(score_dict)
    total_score = 0
    for k, v in score_dict.items():
        total_score += v
    return total_score / num_samples


def traj_success_rate(traj, targets, wall_locations_inner):
    """
    traj: B, horizon, dim=4
    targets: B, dim=2, avoid eval_time (extra dimension)
    wall_locations_inner: direct output from env.wall_locations_inner
    """
    ## 0. preprocess convert to cuda
    traj = utils.to_torch(traj[:,:,:2], device='cuda')
    targets = utils.to_torch(targets, device='cuda')
    wall_locations_inner = utils.to_torch(wall_locations_inner, device='cuda')
    num_samples = traj.shape[0]
    if wall_locations_inner.dim() == 2:
        wall_locations_inner.unsqueeze_(0)
    
    ## traj 1 torch.Size([40, 144, 2]) torch.Size([40, 2])
    # print('traj 1', traj.shape, targets.shape)
    # print('wall_locations_inner', wall_locations_inner)
    
    ## 1. check if valid
    is_valid = maze2d_is_valid_traj(traj, wall_locations_inner) # B,
    traj = traj[is_valid]

    ## 2. check if reach goal
    # print('targets 1', targets.shape)
    targets = targets[is_valid]
    print('number of valid traj:', targets.shape)
    ## -1: last point is set condition
    loc_diff = traj[:, -2, :] - targets
    dist = torch.linalg.norm(loc_diff, ord=None, dim=-1) # B,
    is_reach = dist < 0.5

    cnt_reach = (is_reach).sum().item()
    success_rate = cnt_reach / num_samples

    return success_rate




def vis_reorder_sample(arr, vis_nrow, vis_ncol, samples_perprob):
    """
    make put same trajectories sample in adjacent rows
    arr: numpy array (B, horizon, dim)
    """
    out_shape = arr.shape
    tmp = arr.copy().reshape(vis_nrow, vis_ncol, samples_perprob, *out_shape[1:])
    tmp = np.swapaxes(tmp, axis1=1, axis2=2)
    # print('vis_modelout_path_list 2', tmp.shape)
    arr = tmp.reshape(-1, *out_shape[1:])
    return arr


def wallLoc_totorch_expand(wallLoc, batch_size, device):
    '''
    np: (num_walls, dim) -> tensor: (B, num_walls*dim)
    wallLoc (np.ndarray): num_walls, dim
    '''
    wallLoc = utils.to_torch(wallLoc,  dtype=torch.float32, device=device)
    assert wallLoc.dim() <= 2
    if wallLoc.dim() == 2:
        wallLoc = wallLoc.flatten()
    # print(f'[wallLoc_totorch_expand] wallLoc {wallLoc.shape}') # [3*2,]
    wallLoc = wallLoc.unsqueeze(0).repeat(batch_size, 1)
    return wallLoc


def dyn_wallLoc_b_h_nw_d(wall_c, num_walls, wall_dim):
    '''reshape to (B horizon, nw, dim); horizon auto infer'''
    return einops.rearrange(wall_c, 'b (h n_w d) -> b h n_w d', n_w=num_walls, d=wall_dim)


def sample_pos_compose_maze(base_maze_arr, wall_locations_list):
    w = wall_locations_list
    if type(w) == list: 
        w = np.array(w)
    assert type(w) == np.ndarray
    ## flatten to num_walls, 2
    if w.ndim == 3: 
        w = w.reshape(-1, w.shape[-1])
    ## set not available pos
    base_maze_arr[w[:,0], w[:,1]] = 10


def save_img(savepath: str, img: np.ndarray):
    imageio.imsave(savepath, img)


def plt_img(img, dpi=200):
    import matplotlib.pyplot as plt
    plt.clf(); plt.figure(dpi=dpi); plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow(img)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.show()



def print_color(s, *args, c='r'):
    if c == 'r':
        # print(Fore.RED + s + Fore.RESET)
        print(Fore.RED, end='')
        print(s, *args, Fore.RESET)
    elif c == 'b':
        # print(Fore.BLUE + s + Fore.RESET)
        print(Fore.BLUE, end='')
        print(s, *args, Fore.RESET)
    elif c == 'y':
        # print(Fore.YELLOW + s + Fore.RESET)
        print(Fore.YELLOW, end='')
        print(s, *args, Fore.RESET)
    else:
        # print(Fore.CYAN + s + Fore.RESET)
        print(Fore.CYAN, end='')
        print(s, *args, Fore.RESET)
    
    

def get_time():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def check_arrays_same_shape(arrays):
    '''check if the given list of np array have the same shape'''
    shapes = [arr.shape for arr in arrays]
    return all(shape == shapes[0] for shape in shapes)


def pad_and_concatenate_2d(arrays: list):
    '''
    Args:
        arrays: a list of 2d np
    returns:
        array: a 3d np, each 2d np padded to the max length in arrays
    '''
    if check_arrays_same_shape(arrays):
        return np.array(arrays)
    # Step 1: Find the maximum number of rows among all arrays
    max_rows = max(arr.shape[0] for arr in arrays)
    # Step 2: Pad each array to the maximum number of rows
    padded_arrays = [np.pad(arr, ((0, max_rows - arr.shape[0]), (0, 0)), mode='edge') for arr in arrays]

    # Step 3: Concatenate the padded arrays
    # concatenated_array = np.concatenate(padded_arrays, axis=1)
    array = np.stack(padded_arrays, axis=0)

    return array

