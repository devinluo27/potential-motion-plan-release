import numpy as np


class SearchTree:
    def __init__(self, env, root, dim=2):
        self.states = np.array([root])
        self.parents = [None]
        self.costs = [0.]
        self.in_goal_region = [False]

    def path(self):
        if not np.any(self.in_goal_region):
            return None
        
        # find the one with the lowest cost
        reached_idxs = np.where(self.in_goal_region)
        costs = np.array(self.costs)[reached_idxs]
        current_index = reached_idxs[0][np.argmin(costs)]

        path = []
        cost = 0
        while True:
            path.append(self.states[current_index])
            if current_index==0:
                break
            current_index = self.parents[current_index]
        path.reverse()
        return path

    def rewire_to(self, child_idx, new_parent_idx):
        self.parents[child_idx] = new_parent_idx

    def set_cost(self, idx, new_cost):
        self.costs[idx] = new_cost

    def insert_new_state(self, env, state, parent_idx, done):
        self.states = np.append(self.states, [state], axis=0)
        self.parents.append(parent_idx)
        self.in_goal_region.append(done)

        # Will be updated in post-processing.
        self.costs.append(-1)

        return self.states.shape[0]-1