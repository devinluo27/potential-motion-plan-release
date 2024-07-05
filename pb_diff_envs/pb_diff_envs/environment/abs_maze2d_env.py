import numpy as np

class AbsMaze2DEnv:
    def load(self, **kwargs):
        print(f'load: {kwargs}')
        pass
    
    def unload_env(self):
        print('unload_env')
        pass

    def get_epi_dist(self, prev_pos, new_goal):
        if self.epi_dist_type == 'sum':
            epi_dist = np.abs(new_goal - prev_pos).sum()
        elif self.epi_dist_type == 'norm':
            epi_dist = np.linalg.norm(new_goal - prev_pos)
        else:
            raise NotImplementedError()
        return epi_dist

    def _get_joint_pose(self):
        '''return current pose of the robot'''
        return self.robot.cur_pose
    
    def get_robot_free_pose(self):
        '''return a random (numpy1d) no collision state'''
        return self.robot.sample_free_config()


class AbsStaticMaze2DEnv(AbsMaze2DEnv):
    '''abstraction place holder'''

    def state_fp(self, state):
        return self.robot._state_fp(state)
    
    def edge_fp(self, state, new_state):
        return self.robot._edge_fp(state, new_state)
    


class AbsDynamicMaze2DEnv(AbsMaze2DEnv):
    '''abstraction place holder'''

    def state_fp(self, state, t):
        return self.robot._state_fp_dynamic(self, state, t)

    def edge_fp(self, state, new_state, t_start, t_end):
        return self.robot._edge_fp_dynamic(self, state, new_state, t_start, t_end)
    
