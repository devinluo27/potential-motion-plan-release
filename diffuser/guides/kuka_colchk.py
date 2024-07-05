import diffuser.utils as utils
import numpy as np
import pybullet as p
from pb_diff_envs.utils.kuka_utils_luo import SolutionInterp
import pdb


class KukaCollisionChecker:
    '''
    collision checker for kuka, dual kuka (pybullet-based)
    '''
    def __init__(self, normalizer, interp_density=3) -> None:
        self.sol_interp = SolutionInterp(density=interp_density) # ori 3
        self.normalizer = normalizer
        self.num_collision_check = 0
        self.use_collision_eps = False
        self.collision_eps = 0.5 # dual: 0.5; 7d: 0.5

    def steerTo(self, start, end, env):
        '''
        steerTo and feasibility_check automatically do interpolation
        input two poses, then linear interpolate to form a traj, and check collision
        start, end (torch [1, 7]):
        return:
            bool: True if no collision is found
        '''
        ## build the env first: env is loaded in main function

        assert start.shape == (1,7) or start.shape == (1,14) or start.shape == (1, 2)
        ## build the input, list of np(7,)
        start = np.squeeze(utils.to_np(start), axis=0)
        end = np.squeeze(utils.to_np(end), axis=0)

        ## unnorm before check
        start = self.normalizer.unnormalize(start, 'observations')
        end = self.normalizer.unnormalize(end, 'observations')


        if not self.use_collision_eps:
            traj = self.sol_interp([start, end])
        else:
            traj = np.stack([start, end], axis=0)


        no_collision, check_cnt = self.check_single_traj(traj, env)
        self.num_collision_check += check_cnt

        return no_collision


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
        '''
        just check if the given pos is in collision
        '''
        normed_pos = normed_pos.detach().cpu().squeeze(0).numpy()
        pos = self.normalizer.unnormalize(normed_pos, 'observations')
        env.robot.set_config(pos)

        ## count here
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
        returns:
            True: no collision / False: has collision
            num_collision_check
        '''
        if self.use_collision_eps:
            no_collision, check_cnt = self.check_1_traj_eps(traj, env)

            return no_collision, check_cnt

        assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
        check_cnt = 0
        for i_t in range(len(traj)):
            check_cnt += 1
            env.robot.set_config(traj[i_t])

            if not env.robot.no_collision():
                return False, check_cnt

        return True, check_cnt

    def check_1_traj_eps(self, traj, env):

        tmp_eps = env.robot.collision_eps
        env.robot.collision_eps = self.collision_eps
        assert traj.ndim == 2 and type(traj) == np.ndarray # torch.is_tensor(traj)
        check_cnt = 0

        for i_t in range( len(traj) - 1 ):
            env.robot.collision_check_count = 0
            is_no_col = env.edge_fp( traj[i_t], traj[i_t+1], )
            check_cnt += env.robot.collision_check_count
            if not is_no_col:
                return False, check_cnt

        env.robot.collision_check_count = 0
        env.robot.collision_eps = tmp_eps


        return True & env.state_fp(traj[-1],), check_cnt + 1


    def steerTo_repl(self, start, end, env):
        '''
        input two normed 1d poses, then linear interpolate to form a traj, and check collision
        not update global cnt
        start, end (torch [7,]):
        return:
        bool: True if no collision is found
        '''
        assert start.ndim == 1
        ## build the input, list of np(7,)
        start = utils.to_np(start)
        end = utils.to_np(end)
        ## [Caution] unnorm before check
        traj = self.sol_interp([start, end])
        no_collision, check_cnt = self.check_single_traj(traj, env)

        return no_collision, check_cnt