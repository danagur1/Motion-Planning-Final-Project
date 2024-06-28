import networkx as nx
from sklearn import neighbors

from discopygal.geometry_utils import collision_detection, conversions
from discopygal.solvers.samplers import Sampler_Uniform
from discopygal.solvers import *

from .metrics import *
from .nearest_neighbors import *


class Roadmap(object):
    """
    A class that represents a 2D roadmap, which has the roadmap graph and the collision detection
    for that roadmap.

    :param scene: the scene of the roadmap
    :type scene: :class:`~solvers.Scene`
    :param robot: robot for which we built the roadmap
    :type robot: :class:`~solvers.Robot`
    :param nearest_neighbors: a nearest neighbors algorithm. if None then use sklearn implementation
    :type nearest_neighbors: :class:`~solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param sampler: sampling algorithm/method. if None then use uniform sampling
    :type sampler: :class:`~solvers.samplers.Sampler`
    """
    def __init__(self, scene, robot, nearest_neighbors=None, sampler=None):
        self.scene = scene
        self.robot = robot
        self.collision_detection = collision_detection.ObjectCollisionDetection(self.scene.obstacles, self.robot)

        self.nearest_neighbors = nearest_neighbors
        if self.nearest_neighbors is None:
            self.nearest_neighbors = NearestNeighbors_sklearn()

        self.sampler = sampler
        if self.sampler is None:
            self.sampler = Sampler_Uniform()
        self.sampler.set_scene(self.scene)
        
        self.G = nx.Graph()

        # Generate a cached nearest neighbors (if not already)
        if type(self.nearest_neighbors) is not NearestNeighborsCached:
            self.nearest_neighbors = NearestNeighborsCached(self.nearest_neighbors)
        
    def add_point(self, point):
        """
        Try to add a point to the roadmap.
        If point is invalid - return False.

        :param point: point to add
        :type point: :class:`numpy.array`

        :return: True if point was added
        :rtype: :class:`bool`
        """
        if self.collision_detection.is_point_valid(point):
            self.nearest_neighbors.add_point(point)
            self.G.add_node(point)
            return True
        return False

    def add_random_point(self):
        """
        Sample a random point and try to add it to the roadmap.
        If point is inavlid - return False

        :return: True if point was added
        :rtype: :class:`bool`
        """
        point = self.sampler.sample()
        return self.add_point(point)
    
    def add_edge(self, p, q):
        """
        Get two vertices p, q and try to connect them.
        If we cannot connect them - return False.

        :param p: first point
        :type p: :class:`~numpy.array`
        :param q: second point
        :type q: :class:`~numpy.array`

        :return: True if edge was added
        :rtype: :class:`bool`
        """
        assert (self.G.has_node(p))
        assert (self.G.has_node(q))
        if p == q:
            return False
        edge = np.array([p, q])
        if self.collision_detection.is_edge_valid(edge):
            self.G.add_edge(p, q)
            return True
        return False
    
    def get_points(self):
        """
        Return a list of all landmarks (points) in the roadmap

        :return: list of landmarks
        :rtype: list<:class:`numpy.array`>
        """
        return self.nearest_neighbors.get_points()


def build_probabilistic_roadmap(scene, robot, num_landmarks, k, nearest_neighbors=None, sampler=None):
    """
    Build a probabilistic roadmap for the robot in a given scene.

    :param scene: scene
    :type scene: :class:`solvers.Scene`
    :param robot: robot
    :type robot: :class:`solvers.Robot`
    :param num_landmarks: number of landmarks to sample for the PRM
    :type num_landmarks: :class:`int`
    :param k: number of nearest neighbors to conenct when building PRM
    :type k: :class:`int`
    :param nearest_neighbors: a nearest neighbors algorithm. if None then use sklearn implementation
    :type nearest_neighbors: :class:`solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param sampler: sampling algorithm/method. if None then use uniform sampling
    :type sampler: :class:`solvers.samplers.Sampler`

    :return: roadmap for the given robot
    :type: :class:`Roadmap`
    """
    prm = Roadmap(scene, robot, nearest_neighbors, sampler)

    # Add start and end locations
    start = robot.start
    end = robot.end
    if type(robot) is RobotRod:
        start = start[0]
        end = end[0]
    prm.add_point(start)
    prm.add_point(end)

    # Sample landmarks
    cnt = 0
    while cnt < num_landmarks:
        if prm.add_random_point():
            cnt += 1
    
    # Connect all points to their k nearest neighbors
    for point in prm.get_points():
        neighbors = prm.nearest_neighbors.k_nearest(point, k+1) # k+1 since one of the neighbors is the point itself
        for neighbor in neighbors:
            prm.add_edge(point, neighbor)

    return prm


class TensorRoadmap(object):
    """
    A class that represents a tensor roadmap, which couples a roadmap of each robot
    (which was built seperatly) into one large graph product representing motion of 
    all the robots.
    Note that since for most (nontrivial) cases the explicit roadmap is intractable,
    we generate an interface which allows us to query the tensor roadmap without
    explicitly building it - and also remembering the subgraph of queries we made.
    
    Graphs are reprsented using networkx.
    *Vertices of a roadmap of a single robot are Point_2, while vertices of the
    tensor roadmap are of type Point_d.*

    :param roadmaps: a mapping between each robot and its roadmap
    :type roadmaps: dict<:class:`solvers.Robot`, :class:`networkx.Graph`>
    :param nearest_neighbors: a nearest neighbors algorithm. if None then use sklearn implementation
    :type nearest_neighbors: :class:`solvers.nearest_neighbors.NearestNeighbors` or :class:`None`
    :param metric: a metric for choosing best edge, can be different then the nearest_neighbors metric!
        If None then use euclidean metric
    :type metric: :class:`solvers.metrics.Metric` or :class:`None`
    """
    def __init__(self, roadmaps, nearest_neighbors=None, metric=None):
        self.roadmaps = roadmaps
        
        self.nearest_neighbors = nearest_neighbors
        if self.nearest_neighbors is None:
            self.nearest_neighbors = NearestNeighbors_sklearn()

        # Generate a cached nearest neighbors (if not already)
        if type(self.nearest_neighbors) is not NearestNeighborsCached:
            self.nearest_neighbors = NearestNeighborsCached(self.nearest_neighbors)
        
        self.metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

        self.robots = list(self.roadmaps.keys()) # remeber a *consistent* ordering on the robots
        self.T = nx.Graph() # the subgraph of the tensor roadmap of discovered vertices and edges

    def add_tensor_vertex(self, tensor_point):
        """
        Add explictly a tensor vector to the graph
        (for example when adding start and end points)

        :param tensor_point: a point in the high dimensional space
        :type tensor_point: :class:`numpy.array`
        """
        self.nearest_neighbors.add_point(tensor_point)
        self.T.add_node(tensor_point)

    def nearest_tensor_vertex(self, tensor_point):
        """
        Get a tensor point and return a tensor vertex (i.e. vertex of \hat(V), the vertices of the tensor graph)
        that is the closest when comparing in the high dimensional sapce.

        :param tensor_point: a point in the high dimensional space
        :type tensor_point: :class:`numpy.array`
        """
        if not self.T.has_node(tensor_point):
            return self.nearest_neighbors.k_nearest(tensor_point, 1)[0]
        else:
            # If we query an existing node, we will get it (since dist=0)
            # Hence we want the *second* closest node
            knn = self.nearest_neighbors.k_nearest(tensor_point, 2)
            if len(knn) == 1:
                # If it is the only vertex return it
                return knn[0]
            for point in knn:
                if point != tensor_point:
                    return point


    def clean_tensor_edge(self, p, q):
        """
        Get a tensor edge (made of tensorr vertices p and q) and clean it, i.e.
        make sure that no two robots are intersecting each other.
        If two robots intersect - then make sure one of them stays in place.

        :param p: first vertex
        :type p: :class:`numpy.array`
        :param q: second vertex
        :type q: :class:`numpy.array`

        :return: q', a vertex for which (p,q') is a valid motion
        :rtype: :class:`numpy.array`
        """
        if p == q:
            return None
        for i in range(len(self.robots)):
            for j in range(i+1, len(self.robots)):
                robot1 = self.robots[i]
                robot2 = self.robots[j]
                edge1 = np.array([
                    [p[2*i], p[2*i+1]],
                    [q[2*i], q[2*i+1]]
                ])
                edge2 = np.array([
                    [p[2*j], p[2*j+1]],
                    [q[2*j], q[2*j+1]]
                ])
                if collision_detection.collide_two_robots(robot1, edge1, robot2, edge2):
                    # If they collide, put robot2 in place and try again
                    coords = np.array([])
                    for k in range(q.dimension()):
                        if k // 2 == j:
                            coords.append(p[k])
                        else:
                            coords.append(q[k])
                    q_prime = coords
                    if q == q_prime:
                        return None
                    return self.clean_tensor_edge(p, q_prime)
        return q


    
    def find_best_edge(self, tensor_vertex, tensor_point):
        """
        Given a vertex in the tensor graph and a random tensor point,
        choose an edge in the tensor graph that is made of the edges in the original
        roadmaps that take us the closest to the tensor point.

        Also note that if two robots collide, we try to make one of them to stay in place
        (i.e. some robots might stay in place, but at least one robot is guaranteed to move - if an edge was added).

        :return: True if an edge was added
        :rtype: :class:`bool`
        """   
        vertex = []
        tensor_vertex = conversions.Point_d_to_Point_2_list(tensor_vertex)
        tensor_point = conversions.Point_d_to_Point_2_list(tensor_point)
        for i in range(len(self.robots)):
            assert(self.roadmaps[self.robots[i]].G.has_node(tensor_vertex[i]))
            best_p = None
            best_dist = None
            for edge in self.roadmaps[self.robots[i]].G.edges(tensor_vertex[i]):
                if edge[0] == tensor_vertex[i]:
                    p = edge[1]
                else:
                    p = edge[0]
                dist = self.metric.dist(p, tensor_point[i])
                if best_dist is None or dist < best_dist:
                    best_p = p
                    best_dist = dist
            vertex.append(best_p)
        
        
        vertex = conversions.Point_2_list_to_Point_d(vertex)
        tensor_vertex = conversions.Point_2_list_to_Point_d(tensor_vertex)
        
        # Make sure robots are not colliding
        vertex = self.clean_tensor_edge(tensor_vertex, vertex)
        if vertex is None:
            # If we couldn't connect, then ignore
            return False
        
        self.nearest_neighbors.add_point(vertex)
        self.T.add_node(vertex)
        self.T.add_edge(tensor_vertex, vertex, weight=self.metric.dist(tensor_vertex, vertex).to_double())
        return True

    def try_connecting(self, vertex1, vertex2):
        """
        Try connecting two vertices of the tensor graph. 
        We do that by the method proposed by de Berg:
        http://www.roboticsproceedings.org/rss05/p18.html
        (Finding each path individually and then trying to find a prioritization
        which is collision free).
        Return True if we succeeded connecting.

        :param vertex1: first vertex in the tensor roadmap
        :type vertex1: :class:`numpy.array`
        :param vertex2: second vertex in the tensor roadmap
        :type vertex1: :class:`numpy.array`

        :return: True if we connected the vertices
        :rtype: :class:`bool`
        """
        # Find path for eact robot
        vertex1_points = conversions.Point_d_to_Point_2_list(vertex1)
        vertex2_points = conversions.Point_d_to_Point_2_list(vertex2)
        paths = []
        for i, robot in enumerate(self.robots):
            if not nx.algorithms.has_path(self.roadmaps[robot].G, vertex1_points[i], vertex2_points[i]):
                return False
            path = nx.algorithms.shortest_path(
                self.roadmaps[robot].G, vertex1_points[i], vertex2_points[i])
            paths.append(path)
        
        # Generate the priority graph
        priority_graph = nx.DiGraph()
        priority_graph.add_nodes_from(range(len(self.robots)))
        for i, robot_i in enumerate(self.robots):
            for j, robot_j in enumerate(self.robots):
                if i == j:
                    continue
                # Robot i stays in place in its target
                edge_i = np.array([vertex2_points[i], vertex2_points[i]])

                # Robot j moves along its path
                for k in range(len(paths[j])-1):
                    edge_j = np.array([paths[j][k], paths[j][k+1]])
                    if collision_detection.collide_two_robots(robot_i, edge_i, robot_j, edge_j):
                        # If j collides with i when i is in its target, j needs to reach the target before i 
                        priority_graph.add_edge(j, i)

                # Robot i stays in place in its start
                edge_i = np.array([vertex1_points[i], vertex1_points[i]])

                # Robot j moves along its path again
                for k in range(len(paths[j])-1):
                    edge_j = np.array([paths[j][k], paths[j][k+1]])
                    if collision_detection.collide_two_robots(robot_i, edge_i, robot_j, edge_j):
                        # If j collides with i when i is in its start, i needs to reach the target before j
                        priority_graph.add_edge(i, j)
        
        # If there are cycles, no ordering can be found
        if not nx.algorithms.is_directed_acyclic_graph(priority_graph):
            return False

        # Build a path based on the topological ordering
        ptr = [p for p in vertex1_points] # start at vertex1
        for r in nx.algorithms.topological_sort(priority_graph):
            for v in paths[r]:
                # Advance only the currect robot
                new_ptr = [p for p in ptr]
                new_ptr[r] = v

                self.T.add_edge(
                    conversions.Point_2_list_to_Point_d(ptr),
                    conversions.Point_2_list_to_Point_d(new_ptr)
                )

                ptr = new_ptr
                
        assert(ptr == vertex2_points) # We should have reached the destination
        return True
                
    def get_tensor_subgraph(self):
        """
        Get the subgraph of the tensor roadmap discovered so far
        """
        return self.T
    