import numpy as np
import scipy.interpolate as interpolate
import pdb
import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, path_lengths=None, maze_size=None, eval_solo=False, norm_const_dict={}, kuka_start_end=None):
        """
        dataset: a dict (No a ReplayBuffer class, `field` of the sequencedataset)
        # wall_locactions should be flatten
        eval_solo: set to true when load from serialization.py when eval
        """
        if eval_solo:
            for k, v in dataset.items():
                assert v.ndim == 2
        else:
            dataset = flatten(dataset, path_lengths)

        ## for maze2d observation_dim:4, action_dim:2
        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        ## create a normalizer for each key, limitsNorm by default
        for key, val in dataset.items():
            # print('key:', key); pdb.set_trace()
            try:
                # pdb.set_trace()
                if key == 'infos/wall_locations':
                    assert key not in norm_const_dict.keys()
                    if 'is_sol_kp' in dataset.keys(): ## is_kuka
                        # pdb.set_trace()
                        self.normalizers[key] = Kuka3DWallLocLimitsNormalizer(val, start_end=kuka_start_end)
                    else:
                        self.normalizers[key] = WallLocLimitsNormalizer(val, maze_size=maze_size)
                    print(f'[DatasetNormalizer] key {key}, val {val.shape}') # val is 2D (9912, 6)
                else:
                    if key in norm_const_dict.keys():
                        ## direcly use the given value to normalize
                        assert normalizer == LimitsNormalizer
                        self.normalizers[key] = normalizer(val, norm_const_dict[key])
                    else:
                        self.normalizers[key] = normalizer(val)
                    if key == 'actions':
                        self.normalizers[key].maxs += 1e-7
                        # self.normalizers[key].mins += 1e-7

                # print(f'key {key}, min {self.normalizers[key].mins.shape}') # min is 1D
            except Exception as exce:
                utils.print_color(f'[ utils/normalization ] Skipping {key} | {normalizer}')
                utils.print_color(exce)


    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
        [NOTE] the flatten dataset is trimmed back to path_lengths
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened



#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
        X is flattened -> 
    '''

    def __init__(self, X, min_max=None):
        ''' we need to support given custom min max value.
        min_max: a tuple of np1d (min, max)'''
        self.X = X.astype(np.float32)
        if min_max is not None:
            self.mins = min_max[0]
            self.maxs = min_max[1]
        else:
            ## e.g., [4000, 6] -> [6,]
            self.mins = X.min(axis=0)
            self.maxs = X.max(axis=0)
        # print(f'self.mins: {self.mins}')
        assert self.mins.shape == X.min(axis=0).shape

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()





class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## NOTE mins and maxs are computed on axis=0 only
        ## [B,horizon,correct_dim]
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=0): # 1e-4, might be out of limit
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins


class WallLocLimitsNormalizer(LimitsNormalizer):
    '''
    we do not normalize wall location by default
    '''
    def __init__(self, X, maze_size):
        '''Must override Init because the min & max are different
        maze_size is a tuple, absoulte size
        '''
        assert len(X.shape) == 2, "X must be flatten"
        self.X = X.astype(np.float32)
        ## e.g., [4000, 6] -> [6,]
        self.mins = np.zeros(shape=(X.shape[1],), dtype=np.float32)
        self.mins = self.mins + 1

        self.num_walls = X.shape[1] // 2 # n,8 -> 4 walls
        self.maxs = np.stack([maze_size for _ in range(self.num_walls)], axis=0)
        self.maxs = self.maxs.flatten().astype(np.float32)
        self.maxs = self.maxs - 2
        # pdb.set_trace()
        # TODO NOTE index from 1 to size-2 ?? because we do not use the 
        print(f'[Luo WallLocLimitsNormalizer] mins:{self.mins}, max:{self.maxs}')


class Kuka3DWallLocLimitsNormalizer(LimitsNormalizer):
    '''
    we do not normalize wall location by default
    '''
    def __init__(self, X, start_end):
        '''Must override Init because the min & max are different
        maze_size is a tuple, absoulte size
        '''
        assert len(X.shape) == 2, "X must be flatten"
        assert start_end.shape == (3, 2), 'must be np'
        assert X.shape[1] % 3 == 0

        self.num_walls = X.shape[1] // 3 # n,60 -> 20 walls

        min_single = start_end[:, 0] # (3,)
        max_single = start_end[:, 1] # (3,)
        self.mins = np.stack([min_single for _ in range(self.num_walls)], axis=0) # (20,3)
        self.maxs = np.stack([max_single for _ in range(self.num_walls)], axis=0) # (20,3)

        self.mins = self.mins.flatten().astype(np.float32)
        self.maxs = self.maxs.flatten().astype(np.float32)
        
        # pdb.set_trace()
        # TODO NOTE index from 1 to size-2 ?? because we do not use the 
        print(f'[Luo WallLocLimitsNormalizer] mins:{self.mins}, max:{self.maxs}')





def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

