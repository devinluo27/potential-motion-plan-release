import os
import pickle
import glob
import torch
import pdb
import h5py
from tqdm import tqdm
import numpy as np
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from diffuser.datasets.preprocessing import rm2d_kuka_preproc

from collections import namedtuple

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')


def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)

    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    # print(config)
    return config




def load_datasetNormalizer(h5path, args, train_env_list):
    ''' h5path: path to train dataset
    normalizer: a class
    '''
    assert type(args.normalizer) == str
    is_kuka = 'kuka' in train_env_list.name # hasattr(train_env_list, 'robot_env')
    normalizer = eval(args.normalizer)
    
    # ------------------- load from abc, can be extracted -------------------
    norm_const_dict = args.dataset_config.get('norm_const_dict', False)
    if norm_const_dict:
        data_dict = {}
        data_dict['actions'] = np.array([[0,0], [1,1]], dtype=np.float32)
        data_dict['observations'] = np.array(norm_const_dict['observations'], dtype=np.float32)
        maze_size = None if is_kuka else train_env_list.maze_arr.shape
        d_norm = DatasetNormalizer(data_dict, normalizer, eval_solo=True, maze_size=maze_size)
        return d_norm
    # -------------------


    if is_kuka:
        load_keys = ['observations',] # get_keys(dataset_file)
    else:
        load_keys = ['observations', 'infos/wall_locations'] # get_keys(dataset_file)
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(load_keys, desc="load datafile"):
            try:  # first try loading as an array
                if k != 'observations':
                    data_dict[k] = dataset_file[k][:]
                else:
                    data_dict[k] = dataset_file[k][ ..., args.dataset_config['obs_selected_dim'] ]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    ## infos/wall_locations (2500000, 20, 6)
    ## NOTE dummy placeholder
    data_dict['actions'] = np.array([[0,0], [1,1]], dtype=np.float32)
    '''
    Wall locations shape:
    maze2d: n, n_wall*2
    kuka: n, n_wall, 6
    '''

    if is_kuka:
        if 'infos/wall_locations' in load_keys:
            rm2d_kuka_preproc(data_dict,) # wloc_select=, flatten, 3d -> 2d
        maze_size = None
    else:
        maze_size = train_env_list.maze_arr.shape
    d_norm = DatasetNormalizer(data_dict, normalizer, eval_solo=True, maze_size=maze_size)
    return d_norm

def load_potential_diffusion_model(*loadpath, epoch='latest', device='cuda:0', ld_config={}):
    """
    This function automatically load the objects used in training,
    so to test unseen mazes, we need to modify some objects.
    """
    # dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    trainer_config._dict['results_folder'] = os.path.join(*loadpath)



    render_config._dict['env'] = ld_config['env_instance'] # a env class, save init time


    dataset = RandomNumberDataset(size=1)
    renderer = render_config(is_eval=True)
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)



class RandomNumberDataset(torch.utils.data.Dataset):
    '''
    a placeholder dataset
    '''
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]