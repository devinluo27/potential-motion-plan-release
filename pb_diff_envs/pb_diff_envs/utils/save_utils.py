import numpy as np
import gym, pdb
import os.path as osp
import os
import matplotlib.pyplot as plt

def save_envlist_xyz_idx_list(env_dvg_list, env_name):
    '''aka wall_idx in d4rl
    xyz_idx_list: (num_groups, num_walls)'''
    prefix = './datasets'
    npyname = '%s/%s_xyz_idx_list.npy' % (prefix, env_name)
    is_exist = osp.exists(npyname)
    if is_exist:
        ## check equal
        assert (np.load(npyname) == env_dvg_list.xyz_idx_list).all()
    else:
        np.save(npyname, env_dvg_list.xyz_idx_list)
        os.chmod(npyname, 0o444)
    return

def check_envlist_xyz_idx_list(env_dvg_list):
    ''' check if matched, only when loading train env_list
    aka wall_idx in d4rl
    xyz_idx_list: (num_groups, num_walls)'''
    hdf5_path: str = env_dvg_list.dataset_url
    if hdf5_path is None:
        return # no loading anything...
    ## get pre-saced path
    npyname = hdf5_path.replace('.hdf5', '_xyz_idx_list.npy')
    hdf5_exist = osp.exists( env_dvg_list.dataset_url )
    if hdf5_exist: ## dataset already exists
        assert osp.exists(npyname)
        assert (np.load(npyname) == env_dvg_list.xyz_idx_list).all()
    return

def get_train_xyz_idx_list(env_dvg_list):
    hdf5_path: str = env_dvg_list.dataset_url
    ## get pre-saced path
    npyname = hdf5_path.replace('.hdf5', '_xyz_idx_list.npy')
    return np.load(npyname) 

def save_scatter_fig(points, save_path=None, n_kuka=1):
    # Separate the x and y coordinates for plotting
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    fig, ax = plt.subplots()
    # Create a scatter plot
    ax.scatter(x_coords, y_coords, color='blue', label='Points')
    if n_kuka == 1:
        ax.scatter([0.,], [0.,], color='Red', label='Robots',marker='x' ,s=500)
    elif n_kuka == 2:
        ax.scatter([-0.5, 0.5], [0., 0.], color='Red', label='Robots',marker='x' ,s=500)
    else:
        raise NotImplementedError
    ax.axis('equal')
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('2D Points Visualization')
    # Add grid
    ax.grid(True)
    # Add legend
    legend = ax.legend(loc='upper right')
    for handle in legend.legendHandles:
        handle.set_sizes([100])  # Adjust legend marker size
    if save_path:
        fig.savefig(save_path, dpi=350) # Save the figure as an image file
    # Show the plot
    plt.show()
    plt.clf()
    plt.close(fig)
    return

if __name__ == '__main__':
    import sys; 
    del sys.path[0] # delete the directory where this file is
    sys.path.insert(0, '/oscar/data/csun45/yluo73/robot/lemp')
    print(sys.path)
    import environment
    env_name_list = [
        'kuka7d-ng5hs5k-nv6-se382-vr2-hExt20-v0',
        'kuka7d-ng5hs20k-nv6-se382-vr2-hExt20-v0',
        'kuka7d-ng15ks2k-nv20-se64-vr3-hExt08-v0',
    ]
    for env_name in env_name_list:
        env_dvg_list = gym.make(env_name, load_mazeEnv=False)
        save_envlist_xyz_idx_list(env_dvg_list, env_name)
        env_dvg_list = gym.make(env_name, load_mazeEnv=False)
        check_envlist_xyz_idx_list(env_dvg_list)