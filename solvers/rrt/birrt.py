import networkx as nx
import matplotlib.pyplot as plt

from .. import Robot, RobotDisc, RobotPolygon, RobotRod
from .. import Obstacle, ObstacleDisc, ObstaclePolygon, Scene
from .. import PathPoint, Path, PathCollection

from ..samplers import Sampler, Sampler_Uniform
from ..metrics import Metric, Metric_Euclidean
from ..nearest_neighbors import NearestNeighbors, NearestNeighbors_sklearn, NearestNeighborsCached
from discopygal.bindings import *
from ...geometry_utils import collision_detection, conversions
from ..Solver import Solver


class BiRRT(Solver):
    """
    Implementation of the BiRRT algorithm.
    We grow two trees - one rooted at start, the other at the end. Every once in a while
    (n_join) try to extend both trees toward the same sample.
    Supports multi-robot motion planning, though might be inefficient for more than
    two-three robots.

    :param num_landmarks: number of landmarks to sample
    :type num_landmarks: :class:`int`
    :param eta: maximum distance when steering
    :type eta: :class:`~discopygal.bindings.FT`
    :param n_join: period of iterations when trying to grow the trees towards the same sample
    :type n_join: :class:`int`
    :param nearest_neighbors_start: a nearest neighbors algorithm for the tree grown from start. if None then use sklearn implementation
    :type nearest_neighbors_start: :class:`~discopygal.solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param nearest_neighbors_end: a nearest neighbors algorithm for the tree grown from end. if None then use sklearn implementation
    :type nearest_neighbors_end: :class:`~discopygal.solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param metric: a metric for weighing edges, can be different then the nearest_neighbors metric!
        If None then use euclidean metric
    :type metric: :class:`~discopygal.solvers.metrics.Metric` or :class:`None`
    :param sampler: sampling algorithm/method. if None then use uniform sampling
    :type sampler: :class:`~discopygal.solvers.samplers.Sampler`
    """
    def __init__(self, num_landmarks, eta, n_join, bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, nearest_neighbors_start=None, nearest_neighbors_end=None, metric=None, sampler=None):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.eta = eta
        self.n_join = n_join

        self.nearest_neighbors_start : NearestNeighbors = nearest_neighbors_start
        if self.nearest_neighbors_start is None:
            self.nearest_neighbors_start = NearestNeighbors_sklearn()

        # Generate a cached nearest neighbors (if not already)
        if type(self.nearest_neighbors_start) is not NearestNeighborsCached:
            self.nearest_neighbors_start = NearestNeighborsCached(self.nearest_neighbors_start)

        self.nearest_neighbors_end : NearestNeighbors = nearest_neighbors_end
        if self.nearest_neighbors_end is None:
            self.nearest_neighbors_end = NearestNeighbors_sklearn()

        # Generate a cached nearest neighbors (if not already)
        if type(self.nearest_neighbors_end) is not NearestNeighborsCached:
            self.nearest_neighbors_end = NearestNeighborsCached(self.nearest_neighbors_end)

        self.metric : Metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

        self.sampler : Sampler = sampler
        if self.sampler is None:
            self.sampler = Sampler_Uniform()

        self.roadmap_start = None
        self.roadmap_end = None
        self.roadmap = None
        self.collision_detection = {}
        self.start = None
        self.end = None

    @staticmethod
    def get_arguments():
        """
        Return a list of arguments and their description, defaults and types.
        Can be used by a GUI to generate fields dynamically.
        Should be overridded by solvers.

        :return: arguments dict
        :rtype: :class:`dict`
        """
        return {
            'num_landmarks': ('Number of Landmarks:',1000, int),
            'n_join': ('Expand simultaneously every:', 10, int),
            'eta': ('eta for steering:', 0.1, FT),
            'bounding_margin_width_factor': ('Margin width factor (for bounding box):', Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, FT),
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridded by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return BiRRT(d['num_landmarks'], FT(d['eta']), d['n_join'], FT(d['bounding_margin_width_factor']), None, None, None)

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridded by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or :class:`None`
        """
        return self.roadmap

    def steer(self, p_near, p_rand, eta):
        """
        Steer in eta units from p_near towards p_rand
        """
        dist = self.metric.dist(p_near, p_rand)
        alpha = eta / dist
        alpha = conversions.FT_to_float(alpha)
        if alpha > 1:
            alpha = 1
        d = p_near.dimension()
        coords = []
        for i in range(d):
            coords.append(p_rand[i] * alpha + p_near[i] * (1 - alpha))
        return Point_d(d, coords)

    def collision_free(self, p, q):
        """
        Get two points in the configuration space and decide if they can be connected
        """
        p_list = conversions.Point_d_to_Point_2_list(p)
        q_list = conversions.Point_d_to_Point_2_list(q)

        # Check validity of each edge seperately
        for i, robot in enumerate(self.scene.robots):
            edge = Segment_2(p_list[i], q_list[i])
            if not self.collision_detection[robot].is_edge_valid(edge):
                return False

        # Check validity of coordinated robot motion
        for i, robot1 in enumerate(self.scene.robots):
            for j, robot2 in enumerate(self.scene.robots):
                if j <= i:
                    continue
                edge1 = Segment_2(p_list[i], q_list[i])
                edge2 = Segment_2(p_list[j], q_list[j])
                if collision_detection.collide_two_robots(robot1, edge1, robot2, edge2):
                    return False

        return True

    def sample_free(self):
        """
        Sample a free random point
        """
        p_rand = []
        for robot in self.scene.robots:
            sample = self.sampler.sample()
            while not self.collision_detection[robot].is_point_valid(sample):
                sample = self.sampler.sample()
            p_rand.append(sample)
        p_rand = conversions.Point_2_list_to_Point_d(p_rand)
        return p_rand

    def load_scene(self, scene):
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        # Build collision detection for each robot
        for robot in scene.robots:
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

        ################
        # Build the RRT
        ################
        self.roadmap = nx.Graph()
        self.roadmap_start = nx.Graph()
        self.roadmap_end = nx.Graph()

        curr_tree = self.roadmap_start
        prev_tree = self.roadmap_end
        curr_nn = self.nearest_neighbors_start
        prev_nn = self.nearest_neighbors_end

        # Add start & end points (but only the start to the graph)
        self.start = conversions.Point_2_list_to_Point_d([robot.start for robot in scene.robots])
        self.end = conversions.Point_2_list_to_Point_d([robot.end for robot in scene.robots])
        self.roadmap_start.add_node(self.start)
        self.nearest_neighbors_start.add_point(self.start)
        self.roadmap_end.add_node(self.end)
        self.nearest_neighbors_end.add_point(self.end)

        for cnt in range(self.num_landmarks):
            p_rand = self.sample_free()
            p_near = curr_nn.k_nearest(p_rand, 1)[0]
            p_new = self.steer(p_near, p_rand, self.eta)

            if self.collision_free(p_near, p_new):
                curr_tree.add_edge(p_near, p_new, weight=self.metric.dist(p_near, p_new).to_double())
                curr_nn.add_point(p_new)
                curr_tree, prev_tree = prev_tree, curr_tree
                curr_nn, prev_nn = prev_nn, curr_nn

            # try expanding both trees
            if cnt % self.n_join == 0:
                p_rand = self.sample_free()
                p_near_start = self.nearest_neighbors_start.k_nearest(p_rand, 1)[0]
                p_near_end = self.nearest_neighbors_end.k_nearest(p_rand, 1)[0]
                p_new_start = self.steer(p_near_start, p_rand, self.eta)
                p_new_end = self.steer(p_near_end, p_rand, self.eta)

                if self.collision_free(p_near_start, p_new_start) and self.collision_free(p_near_end, p_new_end):
                    self.roadmap_start.add_edge(p_near_start, p_new_start)
                    self.nearest_neighbors_start.add_point(p_new_start)
                    self.roadmap_end.add_edge(p_near_end, p_new_end)
                    self.nearest_neighbors_end.add_point(p_new_end)

            if cnt % 100 == 0 and self.verbose:
                print('added', cnt, 'landmarks in RRT', file=self.writer)

        # Try connecting both trees
        self.roadmap.add_edges_from(self.roadmap_start.edges)
        self.roadmap.add_edges_from(self.roadmap_end.edges)
        min_d = None
        p_start = None
        p_end = None
        for p in self.roadmap_start.nodes:
            q = self.nearest_neighbors_end.k_nearest(p, 1)[0]
            if min_d is None or self.metric.dist(p, q) < min_d:
                min_d = self.metric.dist(p, q)
                p_start = p
                p_end = q

        if self.collision_free(p_start, p_end):
            self.roadmap.add_edge(p_start, p_end)

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        if not nx.algorithms.has_path(self.roadmap, self.start, self.end):
            if self.verbose:
                print('no path found...', file=self.writer)
            return PathCollection()

        # Convert from a sequence of Point_d points to PathCollection
        tensor_path = nx.algorithms.shortest_path(self.roadmap, self.start, self.end, weight='weight')
        path_collection = PathCollection()
        for i, robot in enumerate(self.scene.robots):
            points = []
            for point in tensor_path:
                points.append(PathPoint(Point_2(point[2*i], point[2*i+1])))
            path = Path(points)
            path_collection.add_robot_path(robot, path)

        if self.verbose:
            print('successfully found a path...', file=self.writer)

        return path_collection
