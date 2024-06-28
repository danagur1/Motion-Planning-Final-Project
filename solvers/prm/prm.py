import networkx as nx
import numpy as np
import time

from .. import Scene
from .. import PathPoint, Path, PathCollection

from ..samplers import Sampler, Sampler_Uniform
from ..metrics import Metric, Metric_Euclidean
from ..nearest_neighbors import NearestNeighbors, NearestNeighbors_sklearn
from geometry_utils import collision_detection, conversions, grid_sdf, hierarchical_sdf, neural_network_sdf
from ..Solver import Solver

CD_METHOD_GRID = "GRID"
CD_METHOD_HIERARCHICAL = "HIERARCHICAL"
CD_METHOD_NN = "NN"


class PRM(Solver):
    """
    The basic implementation of a Probabilistic Road Map (PRM) solver.
    Supports multi-robot motion planning, though might be inefficient for more than
    two-three robots.

    :param num_landmarks: number of landmarks to sample
    :type num_landmarks: :class:`int`
    :param k: number of nearest neighbors to connect
    :type k: :class:`int`
    :param nearest_neighbors: a nearest neighbors algorithm. if None then use sklearn implementation
    :type nearest_neighbors: :class:`~solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param metric: a metric for weighing edges, can be different then the nearest_neighbors metric!
        If None then use euclidean metric
    :type metric: :class:`~solvers.metrics.Metric` or :class:`None`
    :param sampler: sampling algorithm/method. if None then use uniform sampling
    :type sampler: :class:`~solvers.samplers.Sampler`
    """

    def __init__(self, num_landmarks, k, bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 nearest_neighbors=None, metric=None, sampler=None, cd_method="GRID", num_points=1000, edge_samples=100,
                 grid_samples=5, grid_resolution=5, max_depth=3, bbox_samples=5):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.k = k

        self.nearest_neighbors: NearestNeighbors = nearest_neighbors
        if self.nearest_neighbors is None:
            self.nearest_neighbors = NearestNeighbors_sklearn()

        self.metric: Metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

        self.sampler: Sampler = sampler
        if self.sampler is None:
            self.sampler = Sampler_Uniform()

        self.roadmap = None
        self.collision_detection = {}
        self.start = None
        self.end = None
        self.cd_method = cd_method

        #parameters for collision detection
        self.num_points = num_points
        self.edge_samples = edge_samples
        self.grid_samples = grid_samples
        self.grid_resolution = grid_resolution
        self.max_depth = max_depth
        self.bbox_samples = bbox_samples

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
            'num_landmarks': ('Number of Landmarks:', 1000, int),
            'k': ('K for nearest neighbors:', 15, int),
            'bounding_margin_width_factor': (
            'Margin width factor (for bounding box):', Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, np.float64),
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridded by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return PRM(d['num_landmarks'], d['k'], np.float64(d['bounding_margin_width_factor']), None, None, None)

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridded by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or :class:`None`
        """
        return self.roadmap

    def collision_free(self, p, q):
        """
        Get two points in the configuration space and decide if they can be connected
        """
        p_list = conversions.Point_d_to_Point_3_list(p)
        q_list = conversions.Point_d_to_Point_3_list(q)

        # Check validity of each edge seperately
        for i, robot in enumerate(self.scene.robots):
            edge = np.array([p_list[i], q_list[i]])
            if not self.collision_detection[robot].is_edge_valid(edge):
                return False

        # Check validity of coordinated robot motion
        for i, robot1 in enumerate(self.scene.robots):
            for j, robot2 in enumerate(self.scene.robots):
                if j <= i:
                    continue
                edge1 = np.array([p_list[i], q_list[i]])
                edge2 = np.array([p_list[j], q_list[j]])
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
        p_rand = conversions.Point_3_list_to_Point_d(p_rand)
        return p_rand

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~solvers.Scene`
        """
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        # Build collision detection for each robot
        for robot in scene.robots:
            start_time = time.time()
            if self.cd_method == CD_METHOD_GRID:
                val = grid_sdf.ObjectCollisionDetectionGrid(scene.obstacles, robot, edge_samples=self.edge_samples,
                grid_samples=self.grid_samples, grid_resolution=self.grid_resolution)
            if self.cd_method == CD_METHOD_HIERARCHICAL:
                val = hierarchical_sdf.ObjectCollisionDetectionHierarchical(scene.obstacles, robot,
                max_depth=self.max_depth, edge_samples=self.edge_samples, bbox_samples=self.bbox_samples)
            if self.cd_method == CD_METHOD_NN:
                val = neural_network_sdf.ObjectCollisionDetectionNN(scene.obstacles, robot, num_points= self.num_points)
            if not val:
                print("collision detection initialization time 600")
                return False
            self.collision_detection[robot] = val
            end_time = time.time()
            print("collision detection initialization time " + str(end_time-start_time))

        ################
        # Build the PRM
        ################
        self.roadmap = nx.Graph()

        # Add start & end points
        self.start = tuple(conversions.Point_3_list_to_Point_d([robot.start for robot in scene.robots]))
        self.end = tuple(conversions.Point_3_list_to_Point_d([robot.end for robot in scene.robots]))
        self.roadmap.add_node(self.start)
        self.roadmap.add_node(self.end)

        start_time = time.time()
        # Add valid points
        for i in range(self.num_landmarks):
            """if i % 100 == 0:
                print("sampled "+str(i)+ " landmarks")"""
            if time.time()-start_time > 600:
                return False
            p_rand = self.sample_free()
            self.roadmap.add_node(tuple(p_rand))
        end_time = time.time()
        print("sampling time " + str(end_time-start_time))

        self.nearest_neighbors.fit(tuple(self.roadmap.nodes))

        # Connect all points to their k nearest neighbors
        start_time = time.time()
        for cnt, point in enumerate(self.roadmap.nodes):
            neighbors = self.nearest_neighbors.k_nearest(np.array(point), self.k + 1)
            for neighbor in neighbors:
                if self.collision_free(neighbor, np.array(point)):
                    self.roadmap.add_edge(tuple(point), tuple(neighbor), weight=self.metric.dist(point, neighbor))
            if time.time()-start_time > 600:
                break
        end_time = time.time()
        print("roadmap creation time " + str(end_time - start_time))

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~solvers.PathCollection`
        """
        if not nx.algorithms.has_path(self.roadmap, self.start, self.end):
            print('no path found...', file=self.writer)
            return PathCollection()

        # Convert from a sequence of Point_d points to PathCollection
        tensor_path = nx.algorithms.shortest_path(self.roadmap, self.start, self.end, weight='weight')
        path_collection = PathCollection()
        for i, robot in enumerate(self.scene.robots):
            points = []
            for point in tensor_path:
                points.append(PathPoint(
                    [point[2 * i], point[2 * i + 1], point[2 * i + 2], point[2 * i + 3], point[2 * i + 4],
                     point[2 * i + 5]]))
            path = Path(points)
            path_collection.add_robot_path(robot, path)

        print('successfully found a path...', file=self.writer)

        return path_collection
