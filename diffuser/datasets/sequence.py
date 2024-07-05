from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .data_api import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from tqdm import tqdm

Batch = namedtuple('Batch', 'trajectories conditions')
Batch_ml = namedtuple('Batch_ml', 'trajectories conditions wall_locations maze_idx') # maze_idx is an integer


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env=None, horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, dataset_config={}):

        ## --- Faster loading ----
        ## randomize input wallLoc order
        self.env = env = load_environment(env) # 2.would load_env inside
        env.rand_wallorder = dataset_config.get('rand_wallorder', False)
        env.rand_wallorder_seed = dataset_config.get('rand_wallorder_seed', False)
        
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env) # 1.would load_env inside
        # self.env = env = load_environment(env) # 2.would load_env inside
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.maze_change_as_terminal = dataset_config.get('maze_change_as_terminal', False)
        env.maze_change_as_terminal = self.maze_change_as_terminal
        self.cut_episode_len = dataset_config.get('cut_episode_len', False)
        env.cut_episode_len = self.cut_episode_len
        self.noise_injection = dataset_config.get('noise_injection', False)
        self.env_is_eval = env.is_eval
        assert self.noise_injection is None or self.noise_injection <= 1e-2
        print(f'self.noise_injection: {self.noise_injection}, env is_eval: {env.is_eval}')

        # we generate pathes in sequence_dataset
        itr = sequence_dataset(env, self.preprocess_fn)


        # ---------- dataset_config ------------
        # --------------------------------------
        self.is_mazelist = dataset_config.get('is_mazelist', False)
        self.pad_to_horizon = dataset_config.get('pad_to_horizon', False)
        self.ignore_pathlen_0 = dataset_config.get('ignore_pathlen_0', True)
        self.use_normed_wallLoc = dataset_config.get('use_normed_wallLoc', False)
        ## should be tuple maze2d pos (0,1)
        self.obs_selected_dim = dataset_config.get('obs_selected_dim', None)
        self.debug_mode = dataset_config.get('debug_mode', False)

        if 'kuka' in self.env.name:
            self.maze_size = (5,5) ## dummy place holder
            # assert not self.use_normed_wallLoc
        else:
            self.maze_size = self.env.maze_arr.shape



        cnt = 0
        ## select given dim of obs in buffer
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty, self.pad_to_horizon, self.horizon, self.obs_selected_dim)
        # for i, episode in enumerate(itr):
        ## The speed is limited by for loop, not add_path
        for i, episode in enumerate(tqdm(itr)):
            
            if self.ignore_pathlen_0 and episode['observations'].shape[0] == 0:
                continue
            fields.add_path(episode)
            cnt += 1
            
            if self.debug_mode and cnt > 10000:
                break
        
        print('[SequenceDataset] fields Paths cnt:', cnt)
        

        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'], \
            maze_size=self.maze_size, \
            norm_const_dict=dataset_config.get('norm_const_dict', {}), \
            kuka_start_end=getattr(env, 'start_end', None))
        
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        if self.use_normed_wallLoc:
            ## add normed_infos/wall_locations, computed the normed value here
            self.normalize(keys=["infos/wall_locations"])

        print(f'[SequenceDataset] len {self.__len__()}, self.maze_size {self.maze_size}')
        print(fields)


    def normalize(self, keys=['observations', 'actions']):
        '''
            add keys to self.fields
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        print("# number of paths:", len(path_lengths))
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            ## use_padding default to False
            ## max_start is neg if path_length shorter than horizon, so those paths are neglect!
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)

            for start in range(max_start):
                end = start + horizon
                # 1.which path; 2.start; 3.end 
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        # print('path_ind, start, end', path_ind, start, end)

        # observations [horizon, 4]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        ## here already normalized
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        # maze_idx = self.fields['maze_idx']
        # print('maze_idx', maze_idx.shape) # maze_idx (419, 40000, 1)
        maze_idx = self.fields['maze_idx'][path_ind, start]

        if self.use_normed_wallLoc:
            ## load -1~1 normed coordinate e.g.(3,4)
            wall_locations = self.fields['normed_infos/wall_locations'][path_ind, start:end]
            # print('wall_locations', wall_locations)
        else:
            ## load direct coordinate e.g.(3,4)
            wall_locations = self.fields['infos/wall_locations'][path_ind, start:end]

        if self.noise_injection:
            # inplace-add will not change dtype to float64
            trajectories += (np.random.randn(*trajectories.shape) * self.noise_injection)
        # print('maze_idx 2', maze_idx) # shape (1,)
        batch_ml = Batch_ml(trajectories, conditions, wall_locations, maze_idx)
        return batch_ml




class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


        
        


