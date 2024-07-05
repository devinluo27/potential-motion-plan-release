import numpy as np
import pdb

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:
    """
    We add episodes one by one to the ReplayBuffer.
    Example keys:['infos/wall_locations','maze_idx','observations', etc.,]
    """

    def __init__(self, max_n_episodes, max_path_length, termination_penalty, pad_to_horizon=False, horizon=False, obs_selected_dim=None):
        """ Luo's added Parameters: 1. pad_to_horizon::dict ; 2. horizon::int
        pad_to_horizon: pad to horizon len if path is shorter than horizon, 'type': 'copy_last'
        """
        # episode is ~ to one game-play (until finished or out)
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int_),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty
        self.maze_start_idx = []
        self.cur_maze_idx = -1
        self.pad_to_horizon = pad_to_horizon
        self.horizon = horizon
        self.obs_selected_dim = obs_selected_dim


    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def is_obs_and_select(self, key):
        return key =='observations' and \
            self.obs_selected_dim is not None and type(self.obs_selected_dim) != str
    
    def _allocate(self, key, array):
        assert key not in self._dict
        if self.is_obs_and_select(key):
            dim = len(self.obs_selected_dim)
        else:
            dim = array.shape[-1]

        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        """Important function
        one path is a episode from sequence_dataset()
        """
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            ## array should be like (path_len, )
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)

            if self.is_obs_and_select(key):
                self._dict[key][self._count, :path_length] = array[:, self.obs_selected_dim] # 2D
            else:
                self._dict[key][self._count, :path_length] = array

            
        if self.pad_to_horizon and path_length < self.horizon:
            path_length = self.horizon

        
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## 1.check if cross-maze_idx episode exists!
        ## 2. set maze_start_idx: [i] store the first path_idx that in ith maze
        if 'maze_idx' in self.keys:
            maze_idx_0 = self._dict['maze_idx'][self._count, 0]
            # print(self._dict['maze_idx'][self._count])
            ## remember we have padded 0 (max_path_length)
            same_maze_idx = (self._dict['maze_idx'][self._count, :path_length] == maze_idx_0).all()
            if not same_maze_idx:
                print('maze_idx', self._dict['maze_idx'][self._count].shape)
                print("self._dict['maze_idx'][self._count]", self._dict['maze_idx'][self._count])
            assert same_maze_idx, "maze_idx should be the same in one episode!"
            ## [NOTE] be aware that some path_length == 0!!
            if self.cur_maze_idx != maze_idx_0.item() and path_length != 0:
                self.maze_start_idx.append(self._count)
                self.cur_maze_idx = maze_idx_0


        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes, No should be +1 because we terminate at the end??')
