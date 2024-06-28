import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from .. import Robot, RobotPolygon, RobotRod
from .. import Obstacle, ObstacleDisc, ObstaclePolygon, Scene
from .. import PathPoint, Path, PathCollection

from ..samplers import Sampler, Sampler_Uniform
from ..metrics import Metric, Metric_Euclidean
from ..nearest_neighbors import NearestNeighbors, NearestNeighbors_sklearn, NearestNeighborsCached
from ..Solver import Solver
from geometry_utils import collision_detection, conversions, grid_sdf, hierarchical_sdf, neural_network_sdf


CD_METHOD_GRID = "GRID"
CD_METHOD_HIERARCHICAL = "HIERARCHICAL"
CD_METHOD_NN = "NN"


class RRT(Solver):
    """
    Implementation of the plain RRT algorithm.
    Supports multi-robot motion planning, though might be inefficient for more than
    two-three robots.

    :param num_landmarks: number of landmarks to sample
    :type num_landmarks: :class:`int`
    :param eta: maximum distance when steering
    :type eta: :class:`~numpy.float64`
    :param nearest_neighbors: a nearest neighbors algorithm. if None then use sklearn implementation
    :type nearest_neighbors: :class:`~solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param metric: a metric for weighing edges, can be different then the nearest_neighbors metric!
        If None then use euclidean metric
    :type metric: :class:`~solvers.metrics.Metric` or :class:`None`
    :param sampler: sampling algorithm/method. if None then use uniform sampling
    :type sampler: :class:`~solvers.samplers.Sampler`
    """
    def __init__(self, num_landmarks, eta, bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 nearest_neighbors=None, metric=None, sampler=None):
        super().__init__(bounding_margin_width_factor)
        self.num_landmarks = num_landmarks
        self.eta = eta

        self.nearest_neighbors : NearestNeighbors = nearest_neighbors
        if self.nearest_neighbors is None:
            self.nearest_neighbors = NearestNeighbors_sklearn()

        # Generate a cached nearest neighbors (if not already)
        if type(self.nearest_neighbors) is not NearestNeighborsCached:
            self.nearest_neighbors = NearestNeighborsCached(self.nearest_neighbors)

        self.metric : Metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

        self.sampler : Sampler = sampler
        if self.sampler is None:
            self.sampler = Sampler_Uniform()

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
            'eta': ('eta for steering:', 0.1),
            'bounding_margin_width_factor': ('Margin width factor (for bounding box):', Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, FT)
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridded by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return RRT(d['num_landmarks'], (d['eta']), (d['bounding_margin_width_factor']), None, None, None)

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
        alpha = alpha
        if alpha > 1:
            alpha = 1
        d = len(p_near)
        coords = []
        for i in range(d):
            coords.append(p_rand[i] * alpha + p_near[i] * (1 - alpha))
        return np.array(coords)

    def collision_free(self, p, q):
        """
        Get two points in the configuration space and decide if they can be connected
        """
        p_list = p
        q_list = q

        edge = np.array([p[0], q[0]])
        if not self.collision_detection[self.scene.robots[0]].is_edge_valid(edge):
            return False

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


    def load_scene(self, scene):
        super().load_scene(scene)
        self.sampler.set_scene(scene, self._bounding_box)

        # Build collision detection for each robot
        for robot in scene.robots:
            if self.cd_method == CD_METHOD_GRID:
                self.collision_detection[robot] = grid_sdf.ObjectCollisionDetectionGrid(scene.obstacles, robot)
            if self.cd_method == CD_METHOD_HIERARCHICAL:
                self.collision_detection[robot] = hierarchical_sdf.ObjectCollisionDetectionHierarchical(scene.obstacles,
                                                                                                        robot)
            if self.cd_method == CD_METHOD_NN:
                self.collision_detection[robot] = neural_network_sdf.ObjectCollisionDetectionNN(scene.obstacles, robot)

        ################
        # Build the RRT
        ################
        self.roadmap = nx.Graph()

        # Add start & end points (but only the start to the graph)
        self.start = conversions.Point_3_list_to_Point_d([robot.start for robot in scene.robots])
        self.end = conversions.Point_3_list_to_Point_d([robot.end for robot in scene.robots])
        self.roadmap.add_node(self.start)
        self.nearest_neighbors.add_point(self.start)

        for cnt in range(self.num_landmarks):
            p_rand = self.sample_free()
            p_near = self.nearest_neighbors.k_nearest(p_rand, 1)[0]
            p_new = self.steer(p_near, p_rand, self.eta)

            if self.collision_free(p_near, p_new):
                self.roadmap.add_edge(p_near, p_new, weight=self.metric.dist(p_near, p_new))
                self.nearest_neighbors.add_point(p_new)

        # Try adding the target point
        p_new = self.end
        p_near = self.nearest_neighbors.k_nearest(p_new, 1)[0]
        self.roadmap.add_node(p_new)
        if self.collision_free(p_near, p_new):
            self.roadmap.add_edge(p_near, p_new, weight=self.metric.dist(p_near, p_new))


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
                points.append(PathPoint(
                    [point[2 * i], point[2 * i + 1], point[2 * i + 2], point[2 * i + 3], point[2 * i + 4],
                     point[2 * i + 5]]))
            path = Path(points)
            path_collection.add_robot_path(robot, path)

        if self.verbose:
            print('successfully found a path...', file=self.writer)

        return path_collection
