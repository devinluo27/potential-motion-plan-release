from collections import OrderedDict
import numpy as np
from pb_diff_envs.utils.graphs import knn_graph_from_points


from pb_diff_envs.utils.utils import create_dot_dict
from pb_diff_envs.planner.dist_utils import dijkstra
from pb_diff_envs.planner.abstract_planner import AbstractPlanner
import pdb
from colorama import Fore

class SippPlanner(AbstractPlanner):
    def __init__(self, num_samples, k_neighbors=50, stop_when_success=True, eta=1.1):
        self.num_samples = num_samples
        self.stop_when_success = stop_when_success
        self.k_neighbors = k_neighbors

    def _num_node(self):
        ## return nodes on the graph
        return self.num_samples

    def _catch_timeout(self, env, start, goal, timeout, **kwargs):
        if not self.stop_when_success:
            return create_dot_dict(solution=self.search_tree.path())
        else:
            return create_dot_dict(solution=None)

    def create_graph(self):
        graph_data = knn_graph_from_points(self.points, self.k_neighbors)
        self.edges = graph_data.edges
        self.edge_index = graph_data.edge_index
        self.edge_cost = graph_data.edge_cost

    def get_cfg_obs(self):

        time_cfg_obs = OrderedDict()

        max_time_obs = len(self.obs_traj)-1
        # self.speed = 1 / max_time_obs # ori
        self.speed = self.env.speed # Aug 27 added by luo
        ##########
        for i in range(max_time_obs+1):
            time_cfg_obs[i] = self.obs_traj[i]
        return max_time_obs, time_cfg_obs

    def get_safe_intervals(self):
        # safe_intervals {node:[(startTime, endTime), (startTime, endTime),...]}
        safe_intervals = {}

        for mine in range(len(self.points)):
            c_mine = self.points[mine]
            # self.env.robot.set_config(c_mine)
            interval = []
            prev_t = 0
            for time in self.time_cfg_obs:

                c_obs = self.time_cfg_obs[time]

                if not self.env.state_fp(c_mine, time): # collision
                    if time > prev_t:
                        interval.append((prev_t, time-1))
                    prev_t = time + 1


                elif time==next(reversed(self.time_cfg_obs)):
                    interval.append((prev_t, np.inf))

            safe_intervals[mine] = interval
        return safe_intervals


    def setup_planning(self):
        num_objects = len(self.env.objects)

        # [ (x1,2), (x2,2), ] -> [ (x1, 4) ], x1 == x2
        all_traj = np.concatenate([np.array(self.env.objects[i].trajectory.waypoints) for i in range(num_objects)], axis=1)
        all_traj = [all_traj[i] for i in range(all_traj.shape[0])]
        # pdb.set_trace()

        self.obs_traj = all_traj

        self.create_graph()
        self.start_index = 0
        self.goal_index = len(self.points) - 1
        self.max_time_obs, self.time_cfg_obs = self.get_cfg_obs()
        self.safe_intervals = self.get_safe_intervals()
        # print('1:', len(self.safe_intervals[self.start_index]) == 0 )
        # print('2:', self.safe_intervals[self.start_index][0][0] != 0 )

        if len(self.safe_intervals[self.start_index])==0 or self.safe_intervals[self.start_index][0][0] != 0:
            return False
        else:
            return True

    def check_edge_collision(self, src, dest, arr_t, mov_time, src_arr_time):

        mov_start_t = arr_t - mov_time
        disp_mine = self.points[dest] - self.points[src]
        d = np.linalg.norm(disp_mine)
        K = int(np.ceil(d / self.speed))


        # mine waiting
        for time in range(src_arr_time, mov_start_t+1):
            if not self.env.state_fp(self.points[src], min(time, self.max_time_obs)):
                return False

        # print('src:', src, 'dest', dest,)
        # mine moving
        cc = self.env.edge_fp(self.points[src], self.points[dest], mov_start_t, min(mov_start_t + K, self.max_time_obs))

        if not cc:
            return False

        return True

    def get_no_collision_time(self, src, dest, start_t, end_t, interval, mov_time, src_arr_time):
        overlap_s = max(start_t, interval[0])
        overlap_e = min(end_t, interval[1])
        if self.max_time_obs + mov_time <= overlap_s:
            if self.check_edge_collision(src, dest, overlap_s, mov_time, src_arr_time):
                return overlap_s
            else:
                return None

        for arr_t in range(overlap_s, min(overlap_e + 1, self.max_time_obs + mov_time)):
            # check collision on the edge
            if self.check_edge_collision(src, dest, arr_t, mov_time, src_arr_time):
                return arr_t

        return None

    def get_successors(self, edges, src, src_interval_idx, src_arr_time):
        # successor: (node, index of interval, latest non-collision time)
        succs = set()
        for dest in edges[src]:

            mov_time = int(np.ceil(np.linalg.norm(self.points[src] - self.points[dest]) / self.speed))
            start_t = src_arr_time + mov_time
            end_t = self.safe_intervals[src][src_interval_idx][1] + mov_time

            for idx, interval in enumerate(self.safe_intervals[dest]):
                if interval[0] > end_t or interval[1] < start_t:
                    continue
                arr_t = self.get_no_collision_time(src, dest, start_t, end_t, interval, mov_time, src_arr_time)
                if not arr_t:
                    continue
                succ = (dest, idx, arr_t)
                succs.add(succ)
        return succs

    def generatePath(self, arr_time, prev):
        discrete_path = []
        for i in range(len(self.safe_intervals[self.goal_index])):
            goal = (self.goal_index, i)
            if goal in arr_time.keys():
                discrete_path = [(self.points[self.goal_index], arr_time[goal])]
                break

        s = goal
        while prev[s] != (-1,-1):
            discrete_path.append((self.points[prev[s][0]], arr_time[prev[s]]))
            s = prev[s]
        discrete_path.reverse()

        path = []
        for i in range(len(discrete_path)-1):
            path.append(discrete_path[i][0])
            prev_point, prev_t = discrete_path[i][0], discrete_path[i][1]
            next_point, next_t = discrete_path[i+1][0], discrete_path[i+1][1]
            # pdb.set_trace()
            for k in range(1, next_t - prev_t):
                path.append(prev_point + (next_point - prev_point)/np.linalg.norm(next_point - prev_point) * self.speed * k)
        path.append(discrete_path[-1][0])

        return path



    def _plan(self, env, start, goal, timeout, **kwargs):
        '''key function in plan, will be called by plan()'''
        self.env = env
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.points = [self.start] + self.env.robot.sample_n_free_points(self.num_samples) + [self.goal]
        self.points = [np.array(x) for x in self.points]
        if not self.setup_planning():
            print(Fore.RED + 'sipp _plan setup Fail' + Fore.RESET)
            return create_dot_dict(solution=None)


        nodes_idx = list(range(len(self.points)))
        h = dijkstra(nodes_idx, self.edges, self.edge_cost, self.goal_index)

        # node (index of node, index of interval)
        g = {}
        g[(self.start_index, 0)] = 0
        f = {}
        f[(self.start_index, 0)] = h[self.start_index]
        prev = {}
        prev[(self.start_index, 0)] = (-1, -1)

        # earliest non-collision arrival time for every node
        arr_time = {}
        arr_time[(self.start_index, 0)] = 0

        open = set()
        open.add((self.start_index, 0))

        while open:
            s = min(open, key=lambda node: f[node])
            open.remove(s)

            s_index, s_interval_idx = s
            s_arr_t = arr_time[s]
            succs = self.get_successors(self.edges, s_index, s_interval_idx, s_arr_t)
            for item in succs:
                succ_idx, succ_interval_idx, succ_arr_t = item
                succ = (succ_idx, succ_interval_idx)
                if succ not in f.keys():
                    f[succ] = np.inf
                    g[succ] = np.inf
                if g[succ] > succ_arr_t:
                    prev[succ] = s
                    g[succ] = succ_arr_t
                    arr_time[succ] = succ_arr_t
                    f[succ] = g[succ] + h[succ_idx]
                    open.add(succ)
            self.check_timeout(timeout)

        # print('self.goal_index ', self.goal_index )
        # print('arr_time.keys()', arr_time.keys())

        if self.goal_index not in [n[0] for n in arr_time.keys()]:
            return create_dot_dict(solution=None)
        else:
            return create_dot_dict(solution=self.generatePath(arr_time, prev))
