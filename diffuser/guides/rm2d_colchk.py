import torch, time, pdb, einops
import numpy as np
from pb_diff_envs.utils.kuka_utils_luo import SolutionInterp
from pb_diff_envs.objects.trajectory import WaypointLinearTrajectory
from pb_diff_envs.environment.dynamic_rrgroup import DynamicRectangleWall, DynamicRecWallGroup

import diffuser.utils as utils
from .kuka_colchk import KukaCollisionChecker
from pb_diff_envs.utils.maze2d_utils import pad_traj2d_list_v2


class RM2DCollisionChecker(KukaCollisionChecker):
    '''
    only for static maze2d env
    '''
    def __init__(self, normalizer) -> None:
        self.sol_interp = SolutionInterp(density=8) # Sep23: 8; Jan2024: 100
        self.normalizer = normalizer
        self.num_collision_check = 0
        self.use_collision_eps = False
        self.collision_eps = 0.03
        ## steerTo and feasibility_check automatically do interpolation

    def feasibility_check(self, traj, env):
        '''
        traj should be a list of numpy: a *sparse* predicted whole traj from start to goal
        env: a robot env **(should be loaded?)**
        '''
        assert type(traj) == list
        for i in range(0, len(traj)-1):
            ind = self.steerTo(traj[i], traj[i+1], env)
            if not ind:
                return False
        return True
    
    def IsInCollision(self, normed_pos, env):
        '''just check one pos'''
        ## This is not fair to convert to numpy?
        normed_pos = normed_pos.detach().cpu().squeeze(0).numpy()
        pos = self.normalizer.unnormalize(normed_pos, 'observations')
        env.robot.set_config(pos)
        ## count here?
        self.num_collision_check += 1
        if env.robot.no_collision():
            return False
        else:
            return True
    
    def check_single_traj(self, traj, env):
        '''
        check collision of a single traj
        can be operated on torch level so as to save torch->np time
        [NOTE] this method will not update *class checkcount*
        [NOTE] this method accept unnormed traj
        returns:
            True: no collision / False: has collision
            num_collision_check
        '''
        if self.use_collision_eps:
            no_collision, check_cnt = self.check_1_traj_eps(traj, env)
            # pdb.set_trace()
            return no_collision, check_cnt
        
        assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
        check_cnt = 0
        for i_t in range(len(traj)):
            check_cnt += 1
            env.robot.set_config(traj[i_t])
            
            if not env.robot.no_collision():
                return False, check_cnt
            
        return True, check_cnt
    
    ### --- Same in Parent Class, except **edge_fp** ---
    def check_1_traj_eps(self, traj, env):
        '''the goal state is also checked in edge_fp'''
        tmp_eps = env.robot.collision_eps
        env.robot.collision_eps = self.collision_eps
        assert traj.ndim == 2 and type(traj) == np.ndarray
        check_cnt = 0

        for i_t in range( len(traj) - 1 ):
            env.robot.collision_check_count = 0
            is_no_col = env.edge_fp( traj[i_t], traj[i_t+1] )
            check_cnt += env.robot.collision_check_count
            if not is_no_col:
                return False, check_cnt
        
        env.robot.collision_check_count = 0
        env.robot.collision_eps = tmp_eps

        return True, check_cnt
        # return True & env.state_fp(traj[-1]), check_cnt + 1



class DynRM2DCollisionChecker: # (RM2DCollisionChecker)
    def __init__(self, normalizer) -> None:
        self.sol_interp = SolutionInterp(density=8)
        self.normalizer = normalizer
        self.num_collision_check = 0
        self.use_collision_eps = False
        self.collision_eps = 0.03

    def update_wtrajs(self, wtrajs, hExts):
        # assert r_traj.shape[0] == wtrajs.shape[1]
        # wtrajs shape: nw, h, 2
        assert wtrajs.shape[0] <= 10, 'ensure order'
        recWall_list = []
        for i_w in range(len(wtrajs)):
            wpl_traj = WaypointLinearTrajectory(wtrajs[i_w])
            tmp = DynamicRectangleWall(wtrajs[i_w, 0], 
                                        hExts[i_w], wpl_traj)
            recWall_list.append(tmp)
        self.dyn_wgrp = DynamicRecWallGroup(recWall_list)
        self.wtrajs = wtrajs


    def check_single_traj(self, traj, wtrajs):
        '''check collision of a single traj
        can be operated on torch level so as to save torch->np time
        [NOTE] this method will not update *class checkcount*
        [NOTE] this method accept unnormed traj
        returns:
        True: no collision / False: has collision
        num_collision_check
        '''
        if self.use_collision_eps:
            return self.check_1_traj_eps(traj, wtrajs)
        
        assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
        assert traj.shape[0] == self.wtrajs.shape[1]
        assert (self.wtrajs[0] == wtrajs[0]).all(), 'check if correct wtrajs are set'

        check_cnt = 0
        # is_cols = []
        for i_r in range(len(traj)):
            check_cnt += 1
            self.dyn_wgrp.set_walls_at_time(i_r)
            is_col = self.dyn_wgrp.is_point_inside_wg(traj[i_r], min_to_wall_dist=0.01)
            # is_cols.append(is_col)
            if is_col:
                return False, check_cnt
        return True, check_cnt
    

    ### --- Almost Same in Parent Class ---
    def check_1_traj_eps(self, traj, wtrajs):
        '''the goal state is also checked in edge_fp'''
        assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
        assert traj.shape[0] == self.wtrajs.shape[1]
        assert (self.wtrajs[0] == wtrajs[0]).all(), 'check if correct wtrajs are set'

        check_cnt = 0

        for i_t in range( len(traj) - 1 ):
            ## compute displacement
            disp = traj[ i_t+1 ] - traj[ i_t ]
            d = np.linalg.norm( disp )
            K = int(d / self.collision_eps)
            K = K + 1 if i_t == (len(traj) - 2) else K
            # dense_st = []
            for k in range(0, K):
                check_cnt += 1
                cur_state = traj[ i_t ] + k * 1. / K * disp
                # dense_st.append(c)
                # env.robot.set_config( cur_state )
                self.dyn_wgrp.set_walls_at_time(   i_t + k * 1. / K  )
                is_col = self.dyn_wgrp.is_point_inside_wg(cur_state, min_to_wall_dist=0.01)
                # is_cols.append(is_col)
                if is_col:
                    return False, check_cnt

        return True, check_cnt
    
     ### --- Almost Same in Parent Class ---
    def check_1_traj_eps_full_len(self, traj, wtrajs):
        '''the goal state is also checked in edge_fp'''
        assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
        assert traj.shape[0] == self.wtrajs.shape[1]
        assert (self.wtrajs[0] == wtrajs[0]).all(), 'check if correct wtrajs are set'
        cnt, c_list = 0, [] # collision cnt

        for i_t in range( len(traj) - 1 ):
            ## compute displacement
            disp = traj[ i_t+1 ] - traj[ i_t ]
            d = np.linalg.norm( disp )
            K = int(d / self.collision_eps)
            K = K + 1 if i_t == (len(traj) - 2) else K

            for k in range(0, K):
                cur_state = traj[ i_t ] + k * 1. / K * disp
                self.dyn_wgrp.set_walls_at_time(   i_t + k * 1. / K  )
                is_col = self.dyn_wgrp.is_point_inside_wg(cur_state, min_to_wall_dist=0.01)
                if is_col:
                    cnt += 1
                    c_list.append(i_t)
                    break


        return cnt, c_list
    

    def check_single_traj_full_len(self, traj, wtrajs):
        '''
        This function returns: 
        1. * num collisions *, not the collision check count
        2. a list of int, idx of where collision happens

        check collision of a single traj, run the whole traj
        '''
        assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
        assert traj.shape[0] == self.wtrajs.shape[1]
        assert (self.wtrajs[0] == wtrajs[0]).all(), 'check if correct wtrajs are set'

        cnt, c_list = 0, [] # collision cnt
        
        for i_r in range(len(traj)):
            self.dyn_wgrp.set_walls_at_time(i_r)
            is_col = self.dyn_wgrp.is_point_inside_wg(traj[i_r], min_to_wall_dist=0.01)
            if is_col:
                cnt += 1
                c_list.append(i_r)

        return cnt, c_list
    


    
    def reshape_wtrajs(self, wtrajs, hExts, req_len=None):
        # wtrajs np1d shape: h*nw*2
        if wtrajs.ndim == 1:
            # h, nw, dim
            useless = wtrajs.reshape(-1, len(hExts), hExts.shape[1])
            
            # h, nw*d
            wtrajs = wtrajs.reshape(-1, len(hExts) * hExts.shape[1])
            if req_len is not None:
                # h+x, nw*d
                wtrajs = pad_traj2d_list_v2( [wtrajs,], req_len )[0]
            # pdb.set_trace()
            wtrajs = einops.rearrange(wtrajs, 'h (nw d) -> h nw d', nw=len(hExts), d=hExts.shape[1])
            # pdb.set_trace()
            assert (wtrajs[:len(useless)] == useless).all()
            
            # nw, h, dim
            wtrajs = np.transpose(wtrajs, (1, 0, 2) ) # nw, h, dim
        else:
            raise NotImplementedError()
        return wtrajs


# used in eval compute avg
def check_single_dyn_traj(env, traj, wtrajs):
    '''
    traj: robot traj
    wtrajs shape: nw, h, 2
    '''
    checker = DynRM2DCollisionChecker(None)
    wtrajs = checker.reshape_wtrajs(wtrajs, env.wall_hExts, req_len=len(traj)) # nw, 2
    checker.use_collision_eps = True
    # pdb.set_trace()

    checker.update_wtrajs(wtrajs, env.wall_hExts)
    assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)

    if '2DEnv' in repr(env): # or type(env.robot_id) == list: # a robot group
        assert 'Dyn' in repr(env)
        # cnt, c_list = checker.check_single_traj_full_len(traj, wtrajs) # ICLR
        cnt, c_list = checker.check_1_traj_eps_full_len(traj, wtrajs) # Jan 31
    else:
        # env.robot.set_config(traj[i_t])
        raise NotImplementedError()

        
    return cnt, c_list
        