import numpy as np
from pb_diff_envs.planner.search_tree import SearchTree
from pb_diff_envs.utils.utils import create_dot_dict
from pb_diff_envs.planner.abstract_planner import AbstractPlanner
from pb_diff_envs.environment.dynamic_env import DynamicEnv

class RRTStarPlanner(AbstractPlanner):
    
    def __init__(self, stop_when_success=True, steer_eps=None, prob_sample_goal=0.05):
        self.stop_when_success = stop_when_success
        self.steer_eps = steer_eps
        self.prob_sample_goal = prob_sample_goal
        
    def _num_node(self):
        return len(self.search_tree.states)

    def _catch_timeout(self, env, start, goal, timeout, **kwargs):
        if not self.stop_when_success:
            return create_dot_dict(solution=self.search_tree.path())
        else:
            return create_dot_dict(solution=None)

    def _plan(self, env, start, goal, timeout, **kwargs):
        self.search_tree = SearchTree(env=env, root=start, dim=env.robot.config_dim)
        while True:

            if np.random.rand() < self.prob_sample_goal:
                # goal-biased expansion
                leaf_state, parent_idx, no_collision = self.expand(env, goal, sample_state=goal)
            else:
                # uniform expansion
                leaf_state, parent_idx, no_collision = self.expand(env, goal)

            done = env.robot.in_goal_region(leaf_state, goal)

            if no_collision:
                self.search_tree.insert_new_state(env, leaf_state, parent_idx, done)
                self.rewire(env)

            if self.stop_when_success and done:
                break

            self.check_timeout(timeout)

        return create_dot_dict(solution=self.search_tree.path())

    def steer(self, env, sample_state, nearest, dist, steer_eps):
        """Steer the sampled state to a new state close to the search tree.

        Args:
            env: The environment which stores the problem relevant information (map, 
                initial state, goal state), and performs collision check, goal 
                region check, uniform sampling.
            sample_state: State sampled from some distribution.
            nearest: Nearest point in the search tree to the sampled state.
            dist: Distance between sample_state and nearest.

        Returns:
            new_state: Steered state.
        """
        if dist < steer_eps:
            return sample_state

        ratio = steer_eps / dist
        return nearest + (sample_state - nearest) * ratio

    def expand(self, env, goal, sample_state=None):
        """One step of RRT-like expansion.
        """
        tree_states = self.search_tree.states

        # Sample uniformly in the maze
        if sample_state is None:
            sample_state = env.robot.uniform_sample()
            
        # set steer_eps to collision_eps
        if self.steer_eps is None:
            steer_eps = env.robot.collision_eps
        else:
            steer_eps = self.steer_eps            

        # Steer sample to nearby location
        dists = env.robot.distance(tree_states, sample_state)
        nearest_idx, min_dist = np.argmin(dists), np.min(dists)
        new_state = self.steer(env, sample_state, tree_states[nearest_idx], min_dist, steer_eps=steer_eps)

        # no_collision = env.edge_fp(state = tree_states[nearest_idx], new_state=new_state)
        no_collision = self.edge_fp(env, tree_states[nearest_idx], new_state)

        return new_state, nearest_idx, no_collision

    def rewire(self, env, neighbor_r=None):
        """Locally optimize the search tree by rewiring the latest added point.

        Args:
            env: The environment
            neighbor_r (float): Radius for rewiring.
        """
        # set steer_eps to collision_eps
        if self.steer_eps is None:
            steer_eps = env.robot.collision_eps  
        else:
            steer_eps = self.steer_eps

        if neighbor_r is None:
            neighbor_r = steer_eps*3
        cur_tree = self.search_tree.states[:-1]
        new_state = self.search_tree.states[-1]
        nearest = self.search_tree.parents[-1]

        # Find the locally optimal path to the root for the latest point.
        dists = env.robot.distance(cur_tree, new_state)
        near = np.where(dists < neighbor_r)[0]

        min_cost = dists[nearest] + self.search_tree.costs[nearest]
        min_j = nearest
        for j in near:
            cost_new = dists[j] + self.search_tree.costs[j]
            if cost_new < min_cost:
                # if env.edge_fp(state = cur_tree[j], new_state = new_state):
                if self.edge_fp( env, cur_tree[j], new_state):
                    min_cost, min_j = cost_new, j

        # Rewire (change parent) to the locally optimal path.
        self.search_tree.rewire_to(len(self.search_tree.states)-1, min_j)
        self.search_tree.set_cost(len(self.search_tree.states)-1, min_cost)

        # If the latest point can improve the cost for the neighbors, rewire them.
        for j in near:
            cost_new = min_cost + dists[j]
            if cost_new < self.search_tree.costs[j]:
                # if env.edge_fp(state = cur_tree[j], new_state = new_state):
                if self.edge_fp( env, cur_tree[j], new_state):
                    self.search_tree.set_cost(j, cost_new)
                    self.search_tree.rewire_to(j, len(self.search_tree.states)-1)
    
    def edge_fp(self, env, state, new_state):
        if isinstance(env, DynamicEnv):
            tmp = env.edge_fp(state, new_state, 0, 0)
        else:
            tmp = env.edge_fp(state, new_state)
        return tmp
