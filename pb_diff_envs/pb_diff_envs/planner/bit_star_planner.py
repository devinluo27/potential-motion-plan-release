import numpy as np
from pb_diff_envs.utils.utils import create_dot_dict
from scipy import special
from pb_diff_envs.planner.abstract_planner import AbstractPlanner
import math
import heapq
from pb_diff_envs.environment.dynamic_env import DynamicEnv
from gym.utils import seeding
import pdb

INF = float("inf")


class BITStarPlanner(AbstractPlanner):
    def __init__(self, num_batch, stop_when_success=True, eta=1.1, seed=None):
        self.num_batch = num_batch
        self.stop_when_success = stop_when_success

        self.batch_size = num_batch
        self.eta = eta  # a parameter to determine the sampling radius after a solution is found and needs to keep being refined
        assert seed is None or type(seed) == int
        self.seed_input = seed
        self.reset_np_rng()



    def _num_node(self):
        return len(self.samples)
    
    def reset_np_rng(self, trial_idx=0):
        '''different trial should have different randomness'''
        tmp_seed = None if self.seed_input is None else self.seed_input + trial_idx
        self.np_random, self.seed = seeding.np_random(tmp_seed)

        tmp_seed = None if self.seed_input is None else self.seed_input + 10 + trial_idx
        self.robot_np_random, self.robot_seed = seeding.np_random(tmp_seed)



    def setup_planning(self):
        ## make sure every plan's sampling has the same order
        self.reset_np_rng()

        ## 14d below two lines checked
        self.ranges = self.env.robot.limits_high - self.env.robot.limits_low
        self.dimension = self.env.robot.config_dim
        
        self.vertices = []
        self.edges = dict()  # key = point，value = parent
        
        # This is the tree
        self.samples = []  # samples are not included in the tree nodes
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()        
        self.g_scores = dict()

        self.r = INF
        self.n_collision_points = 0
        self.n_free_points = 2        

        # add goal to the samples
        self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        # add start to the tree
        self.vertices.append(self.start)
        self.g_scores[self.start] = 0

        # Computing the sampling space
        self.informed_sample_init()
        radius_constant = self.radius_init()

        return radius_constant

      
    def radius_init(self):
        # Hypersphere radius calculation
        n = self.dimension
        unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
        volume = np.abs(np.prod(self.ranges)) * self.n_free_points / (self.n_collision_points + self.n_free_points)  # the volume of free configuration space
        gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
        radius_constant = 2 * self.eta * (gamma ** (1.0 / n))
        return radius_constant

    def informed_sample_init(self):
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = np.array([(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)])
        a_1 = (np.array(self.goal) - np.array(self.start)) / self.c_min
        
        id1_t = np.array([1.0] * self.dimension)
        
        M = np.dot(a_1.reshape((-1, 1)), id1_t.reshape((1, -1)))
        U, S, Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(np.dot(U, np.diag([1] * (self.dimension - 1) + [np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)

    def sample_unit_ball(self):
        u = self.np_random.normal(0, 1, self.dimension)  # an array of d normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        r = self.np_random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def informed_sample(self, c_best, sample_num, vertices):
        if c_best < float('inf'):
            c_b = math.sqrt(c_best ** 2 - self.c_min ** 2) / 2.0
            r = [c_best / 2.0] + [c_b] * (self.dimension - 1)
            L = np.diag(r)
        sample_array = []
        cur_num = 0
        while cur_num < sample_num:
            if c_best < float('inf'):
                x_ball = self.sample_unit_ball()
                random_point = tuple(np.dot(np.dot(self.C, L), x_ball) + self.center_point)
            else:
                random_point = self.get_random_point()
            if self.is_point_free(random_point):
                sample_array.append(random_point)
                cur_num += 1

            self.check_timeout(self.timeout) # NEW
        return sample_array

    def get_random_point(self):
        point = self.env.robot.uniform_sample(rng=self.robot_np_random)
        return tuple(point)

    def is_point_free(self, point):
        if isinstance(self.env, DynamicEnv):
            result = self.env.state_fp(np.array(point), 0)
        else:
            result = self.env.state_fp(np.array(point))
        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result

    def is_edge_free(self, edge):
        if isinstance(self.env, DynamicEnv):
            result = self.env.edge_fp(np.array(edge[0]), np.array(edge[1]), 0, 0)
        else:
            result = self.env.edge_fp(np.array(edge[0]), np.array(edge[1]))
        return result

    def get_g_score(self, point):
        # gT(x)
        if point == self.start:
            return 0
        if point not in self.edges:
            return INF
        else:
            return self.g_scores.get(point)

    def get_f_score(self, point):
        # f^(x)
        return self.heuristic_cost(self.start, point) + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        if not self.is_edge_free([point1, point2]):
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        # Euler distance as the heuristic distance
        return self.distance(point1, point2)

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_edge_value(self, edge):
        # sort value for edge
        return self.get_g_score(edge[0]) + self.heuristic_cost(edge[0], edge[1]) + self.heuristic_cost(edge[1], self.goal)

    def get_point_value(self, point):
        # sort value for point
        return self.get_g_score(point) + self.heuristic_cost(point, self.goal)

    def bestVertexQueueValue(self):
        if not self.vertex_queue:
            return INF
        else:
            return self.vertex_queue[0][0]

    def bestEdgeQueueValue(self):
        if not self.edge_queue:
            return INF
        else:
            return self.edge_queue[0][0]

    def prune_edge(self, c_best):
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point)

    def prune(self, c_best):
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best]
        self.prune_edge(c_best)
        vertices_temp = []
        for point in self.vertices:
            if self.get_f_score(point) <= c_best:
                if self.get_g_score(point) == INF:
                    self.samples.append(point)
                else:
                    vertices_temp.append(point)
        self.vertices = vertices_temp

    def expand_vertex(self, point):

        # get the nearest value in vertex for every one in samples where difference is less than the radius
        neigbors_sample = []
        for sample in self.samples:
            if self.distance(point, sample) <= self.r:
                neigbors_sample.append(sample)
                
        # add an edge to the edge queue is the path might improve the solution
        for neighbor in neigbors_sample:
            estimated_f_score = self.heuristic_cost(self.start, point) + \
                                self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
            if estimated_f_score < self.g_scores[self.goal]:
                heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        # add the vertex to the edge queue
        if point not in self.old_vertices:
            neigbors_vertex = []
            for ver in self.vertices:
                if self.distance(point, ver) <= self.r:
                    neigbors_vertex.append(ver)
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = self.heuristic_cost(self.start, point) + \
                                        self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
                    if estimated_f_score < self.g_scores[self.goal]:
                        estimated_g_score = self.get_g_score(point) + self.heuristic_cost(point, neighbor)
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

    def get_best_path(self):
        path = []
        if self.g_scores[self.goal] != INF:
            path.append(self.goal)
            point = self.goal
            while point != self.start:
                point = self.edges[point]
                path.append(point)
            path.reverse()
        return list(np.array(path)) if len(path) else None

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length

    def _plan(self, env, start, goal, timeout, **kwargs):
        
        self.env = env
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.timeout = timeout # dict, shallow copy
        self.setup_planning()

        while True:
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.g_scores[self.goal]
                self.prune(c_best)
                self.samples.extend(self.informed_sample(c_best, self.batch_size, self.vertices))
                
                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(point), point) for point in self.vertices]
                heapq.heapify(self.vertex_queue)  # change to op priority queue
                q = len(self.vertices) + len(self.samples)
                self.r = self.radius_init() * ((math.log(q) / q) ** (1.0 / self.dimension))

            try:
                while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                    _, point = heapq.heappop(self.vertex_queue)
                    self.expand_vertex(point)
                    self.check_timeout(timeout) # NEW
            except Exception as e:
                if (not self.edge_queue) and (not self.vertex_queue):
                    continue
                else:
                    raise e

            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)

            # Check if this can improve the current solution
            if best_edge_value < self.g_scores[self.goal]:
                actual_cost_of_edge = self.actual_edge_cost(bestEdge[0], bestEdge[1])
                actual_f_edge = self.heuristic_cost(self.start, bestEdge[0]) + actual_cost_of_edge + self.heuristic_cost(bestEdge[1], self.goal)
                if actual_f_edge < self.g_scores[self.goal]:
                    actual_g_score_of_point = self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        self.g_scores[bestEdge[1]] = actual_g_score_of_point
                        self.edges[bestEdge[1]] = bestEdge[0]
                        if bestEdge[1] not in self.vertices:
                            self.samples.remove(bestEdge[1])
                            self.vertices.append(bestEdge[1])
                            heapq.heappush(self.vertex_queue, (self.get_point_value(bestEdge[1]), bestEdge[1]))

                        self.edge_queue = [item for item in self.edge_queue if item[1][1] != bestEdge[1] or \
                                           (self.get_g_score(item[1][0]) + self.heuristic_cost(item[1][0], item[1][1])) < self.get_g_score(item[1][0])]
                        heapq.heapify(self.edge_queue)  # Rebuild the priority queue because it will be destroyed after the element is removed

            else:
                self.vertex_queue = []
                self.edge_queue = []

            if (self.stop_when_success) and self.g_scores[self.goal] < float('inf'):
                break

            self.check_timeout(timeout)
        
        return create_dot_dict(solution=self.get_best_path())
