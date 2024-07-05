import numpy as np
import pdb, torch
from diffuser.utils import to_torch, to_np
from diffuser.guides.kuka_plan_utils import check_start_end_repl, check_single_traj, check_start_end_repl_v2
from diffuser.guides.kuka_colchk import KukaCollisionChecker
from diffuser.models.diffusion_pb import GaussianDiffusionPB
from typing import Union
from diffuser.guides.policies_compose import PolicyCompose # composed replan to be added

class KukaDiffusionReplanner:
    '''
    replan a given trajectory
    '''
    def __init__(self, model, normalizer, repl_config: dict, device) -> None:
        self.model: Union[GaussianDiffusionPB, PolicyCompose] = model
        ## replan a longer area
        self.gap = 3 # [..., gap, collision_st, ..., collision_end, gap, ...]
        ## the input to model.replan need to be a tenosr
        self.use_ddim = repl_config['use_ddim']
        self.dn_steps = torch.full((1,), fill_value=repl_config['dn_steps'], device=device, dtype=torch.long)
        self.n_replan_trial = repl_config.get('n_replan_trial', 5)
        self.normalizer = normalizer
        self.device = device
        ## interpolation density of collision check
        if normalizer.observation_dim <= 3: # maze2d
            self.collision_eps = 0.05
        else:
            self.collision_eps = 0.5


    def replan_collision_section_less_check(self, env, traj, walls_loc):
        '''call from outside, more efficient impl
        NOTE still have redundant collision check, due to the first full horizon check
        but the traj must be problematic
        traj: np 2D (H, 7/14)
        walls_loc: now input is cpu tensor
        Returns:
            new traj:
            if success after replan:
            num_check:
        '''
        self.num_colck = 0 # of one replan problem
        c_cnt, c_list = check_single_traj(env, traj) ## traj must be unnormed
        assert c_cnt > 0
        # check the whole traj, can be even reduce by some means
        self.num_colck += len(traj) # get where collision happens

        ## A. find out the collision pose idx
        c_list = split_sorted_array_into_subarrays(c_list) # list of np1d

        ## B. expand the collision area for replan
        for i in range(len(c_list)):
            st = c_list[i][0]
            end = c_list[i][-1] # inclusive [s, e]
            c_list[i] = self.get_expanded_subtraj(len(traj), st, end)


        new_traj = traj # deepcopy? seems like no need.
        ## move to device
        if walls_loc.device != self.device:
            walls_loc = walls_loc.to(self.device)
        
        for _ in range(self.n_replan_trial):
            ## a np2d, a bool
            new_traj, is_success, c_list = self.replan_one_time_less_check(env, new_traj, walls_loc, c_list)
            if is_success:
                break
        
        num_colck = self.num_colck
        self.num_colck = None
        return new_traj, is_success, num_colck






    def replan_one_time_less_check(self, env, traj, walls_loc, c_list):
        '''
        One Replan Trial
        ** here for now, the input is unnormed np **
        ** here for now, the output should also be unnormed np **
        Args:
            traj: now np (horizon, 7)
            walls_loc: 1,60
        Returns:
            new_traj:
            is_success:
            c_list_after: a new list of int, indicating the idx of collision after replan

        1. maybe here we can directly operate on cuda tensor to speed up
        2. make it recursive to reduce the number of collision check... now many extra checks
        '''
        assert traj.ndim == 2 and walls_loc.ndim == 2 and walls_loc.shape[0] == 1
       

        ## C. normalize and to cuda tensor for diffusion
        ## slow, very trial needs a np->tensor
        if not torch.is_tensor(traj):
            normed_traj = self.preproc_traj(traj)

        # print('c_list', c_list)

        ## D. replan, get a new list, two impl
        new_section_list = self.replan_whole_traj(c_list, normed_traj, walls_loc)
        
        '''can be speed up by reducing conversion to numpy'''
        ## E. check if the replanned traj is good
        thres = get_se_thres(traj)
        new_traj = to_np(traj) # must be unnormed, np.copy(traj) # torch.clone(traj)
        n_valid = 0 # how many trajs are replaced
        c_list_after = [] # delete from list might be buggy
        for i in range(len(c_list)):
            ## 1. to np and unnormed before check and save
            new_section = new_section_list[i] # cuda tensor
            new_section = to_np(new_section) # (x, 7/14)?
            new_section = self.normalizer.unnormalize(new_section, 'observations')

            ## 2. check if the new_traj is good and accept it
            c1 = check_start_end_repl(new_section, traj[ c_list[i][0] ], traj[ c_list[i][-1] ], thres)
            c2, c_cnt2 = check_start_end_repl_v2(traj, new_section, c_list[i], self.checker, env)
            # NOTE we might need a steerTo like: def check_new_sec_connect(self, s_idx, e_idx):
            self.num_colck += c_cnt2

            # no need to check if above is invalid
            if c1 or c2:            
                no_collision, c_cnt = self.checker.check_single_traj(new_section, env)
                self.num_colck += c_cnt
            
            ## 3. replace if success; else add to list
            if (c1 or c2) and no_collision:
                new_traj[ c_list[i] ] = new_section # to check
                n_valid += 1
            else:
                # a list ot hold the sections still have collision
                c_list_after.append( c_list[i] )


        is_success = (n_valid == len(c_list))
        if is_success: # sanity check
            assert len(c_list_after) == 0

        return new_traj, is_success, c_list_after






    
    def replan_whole_traj(self, c_list, normed_traj, walls_loc):
        '''input the full horizon to diffusion model, 
        and only return a list of new sub-traj of the problematic region (normed)
        '''
        # 0: should be 1, dim
        cond = {0: normed_traj[None, 0], len(normed_traj)-1: normed_traj[None, -1]}
        if self.use_ddim:
            new_traj = self.model.ddim_replan(normed_traj, cond, walls_loc, self.dn_steps)
        else: # ddpm
            new_traj = self.model.replan(normed_traj, cond, walls_loc, self.dn_steps)
        
        new_section_list = []
        for i in range(len(c_list)):
            # pdb.set_trace()
            new_section_list.append(  new_traj[c_list[i]] )
        return new_section_list



    def preproc_traj(self, unnorm_traj):
        '''should be removed for speed later
        input a np, return a new tensor
        1. normalize; 2. to_torch
        '''
        normed_traj = self.normalizer.normalize(unnorm_traj, 'observations')
        return to_torch(normed_traj, device=self.device) 

    def get_expanded_subtraj(self, traj_len, st, end, div_factor=4):
        '''
        return a np array of consecutive number, its len is divisible by 4
        we also need to expand if can directly be divisible
        traj_len: horizon
        '''
        flag = True # expand one end each time
        input_len = end - st + 1
        subtraj_len = end - st + 1
        ## input horizon must be divisible by 4; 
        # tailor traj shape for the cnn model
        while subtraj_len % div_factor != 0 or subtraj_len <= div_factor \
            or subtraj_len <= input_len:
            if flag:
                st = max(st - 1, 0)
                flag = False
            else:
                end = min(end + 1, traj_len - 1)
                flag = True
            subtraj_len = end - st + 1
        
        return np.arange(st, end+1) # inclusive
    
    def set_collision_checker(self, checker: KukaCollisionChecker):
        self.checker = checker


def split_sorted_array_into_subarrays(arr):
    '''
    arr: a list or np1d
    e.g. [1, 2, 5, 6, 7, 8, 9, 10] 
            -> [array([1, 2]), array([ 5,  6,  7,  8,  9, 10])]
    return:
    a list of np array'''
    subarrays = []
    current_subarray = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            current_subarray.append(arr[i])
        else:
            subarrays.append(np.array(current_subarray))
            current_subarray = [arr[i]]

    subarrays.append(np.array(current_subarray))
    return subarrays

def merge_overlapping_arrays(arr_list):
    '''given a list of np1d, return a new list of np1d that merged array with overlap numbers'''
    if not arr_list:
        return []

    merged_list = []
    current_array = arr_list[0]

    for arr in arr_list[1:]:
        if arr[0] <= current_array[-1] + 1:
            current_array = np.concatenate((current_array, arr))
        else:
            merged_list.append(np.unique(current_array))  # Remove duplicates here
            current_array = arr

    tmp = np.sort(np.unique(current_array)) # sort seems to be not necessary
    merged_list.append(tmp)  # Remove duplicates for the last array
    return merged_list


def get_se_thres(traj):
    '''get threshold, replan section should be close enough to original traj'''
    if traj.shape[-1] == 2:
        return 0.15
    elif traj.shape[-1] == 7:
        return 0.4
    elif traj.shape[-1] == 14:
        return 0.4
    else:
        raise NotImplementedError()