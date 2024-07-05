import os
import collections
import numpy as np
import gym, pdb, traceback, copy

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name, **kwargs):
    '''
    would be call several times: 2 in dataset, 1 in rendering
    '''
    if type(name) != str:
        ## we assume name is already an env here, so directly return name
        ## name is already an environment
        assert hasattr(name, 'name')
        return name # copy.deepcopy(name)
    
    ## otherwise, load the env
    wrapped_env = gym.make(name, load_mazeEnv=False, **kwargs)
    
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name

    return env


def get_dataset(env):
    dataset = env.get_dataset()
    return dataset


def sequence_dataset(env, preprocess_fn, prev_last_as_start=False):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env) # a dict, with elem as numpy
    dataset = preprocess_fn(dataset)

    N = dataset['observations'].shape[0] # rewards
    
    ## defaultdict here return a list if missing key
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    print('[ datasets/api ] use_timeouts', use_timeouts) # True

    has_metadata = False
    for k in dataset:
        has_metadata =  has_metadata | ('metadata' in k)
    cut_episode_len = env.cut_episode_len
    
    if cut_episode_len:
        assert env.maze_change_as_terminal, 'must use with the flag set.'
    
    if cut_episode_len: # further cut the episode shorter
        print("dataset['maze_change_bool']", len(dataset['maze_change_bool']))
        assert (len(dataset['maze_change_bool']) % cut_episode_len == 0) and cut_episode_len > 50
        dataset['maze_change_bool'][::cut_episode_len] = True
    
    is_kuka = 'pos_reset' in dataset.keys()
    ignore_set = {'reach_max_episode_steps', 'timeouts', 'infos/goal',} # not actions

    if is_kuka: 
        ignore_set.add('rewards')
    dataset['timeouts'] = dataset['timeouts'].astype(bool) # fit bit operation below
    episode_step = 0
    
    for i in range(N):
        
        # --------- set split path signal -----------
        # done=True: 1.task solved; 2.max len
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts: # default True    
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        
        if env.maze_change_as_terminal:
            done_bool = dataset['maze_change_bool'][i]
            final_timestep = dataset['maze_change_bool'][i]
        else:
            raise ValueError('not supported')


        if is_kuka:
            ## must cut if the pos is forcely reset to 0
            done_bool = done_bool | dataset['pos_reset'][i]
            final_timestep = final_timestep | dataset['pos_reset'][i]

        # assert done_bool == final_timestep # checked
        # --------- set split path signal ends -----------

        # append the sample to data_ (buffer for episode)
        for k in dataset: # iterate keys
            if 'metadata' in k: 
                print('[ datasets/api ] metadata', k)
                continue
            if k in ignore_set: # ignore some of uesless data to speed up
                continue
            ## append last pos as the start pos for this episode
            if prev_last_as_start: # default must be False
                # empty & not the first pos in an env (-1 True)
                if k not in data_.keys() and not dataset['maze_change_bool'][i-1]:
                    data_[k].append(dataset[k][i-1])

            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            # yield here is similar to 'return', last (incomplete) path is not included
            # print('actions', episode_data['actions'].shape)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


