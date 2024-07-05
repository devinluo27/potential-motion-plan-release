from abc import ABC, abstractmethod
from time import perf_counter

from pb_diff_envs.environment.static_env import StaticEnv
from pb_diff_envs.environment.dynamic_env import DynamicEnv
from pb_diff_envs.environment.abs_maze2d_env import AbsStaticMaze2DEnv, AbsDynamicMaze2DEnv
from pb_diff_envs.utils.utils import DotDict, create_dot_dict, print_color


class TimeOutException(Exception):
    def __init__(self, message):
        super(TimeOutException, self).__init__(message)


class AbstractPlanner(ABC):

    def plan(self, env, start, goal, timeout, **kwargs):
        '''
        return an instance of DotDict with:
        1. solution: a list of waypoints. if there is no solution found, the value is None
        2. running_time: the overall running time
        3. num_collision_check: the number of collision checking during the planning
        4. num_node: the number of sampled nodes
        '''
        ## [NOTE] sanity check here will also count to the collision checks
        if isinstance(env, StaticEnv):
            assert env.state_fp(start) and env.state_fp(goal)
        elif isinstance(env, DynamicEnv):
            assert env.state_fp(start,0) and env.state_fp(goal,-1)
        elif isinstance(env, AbsStaticMaze2DEnv):
            assert env.state_fp(start) and env.state_fp(goal)
        elif isinstance(env, AbsDynamicMaze2DEnv):
            assert env.state_fp(start,0) and env.state_fp(goal,-1)

        self.t0 = perf_counter()
        try:
            result = self._plan(env, start, goal, timeout, **kwargs)
        except TimeOutException:
            print_color( f"Planner timeout: {timeout[1]}s." )
            result = self._catch_timeout(env, start, goal, timeout, **kwargs)
        assert isinstance(result, DotDict)
        assert 'solution' in result.keys()
        result.running_time = perf_counter() - self.t0
        result.num_collision_check = env.robot.collision_check_count
        
        env.robot.collision_check_count = 0
        result.num_node = self._num_node()
        return result

    def check_timeout(self, timeout):
        '''
        a function that needs to be called consistently during the planning
        ensure that the planner will be terminated when timeout happens
        '''
        if timeout[0] == 'time':
            if (perf_counter() - self.t0) > timeout[1]:
                raise TimeOutException("Timeout - planner fails to find a solution.")
            else:
                return False
        elif timeout[0] == 'node':
            if self._num_node() > timeout[1]:
                raise TimeOutException("Timeout - planner fails to find a solution.") 
            else:
                return False            

    @abstractmethod
    def _plan(self, env, start, goal, timeout, **kwargs):
        '''
        return an instance of DotDict with:
        1. solution: a list of waypoints. if there is no solution found, the value is None
        '''
        raise NotImplementedError
        
    @abstractmethod
    def _num_node(self):
        '''
        return the number of sampled nodes
        '''
        raise NotImplementedError

    def _catch_timeout(self, env, start, goal, timeout, **kwargs):
        '''
        return an instance of DotDict
        '''
        return create_dot_dict(solution=None)