import numpy as np
import sklearn.neighbors

from .metrics import *


class NearestNeighbors(object):
    """
    Abstract class represeting the interface a nearest neighbors algorithm should comply.

    :param metric: metric to compute nearest neighbors
    :type metric: :class:`~discopygal.solvers.metrics.Metric`
    """
    def __init__(self, metric):
        self.metric = metric
        self.points = []

    def get_points(self):
        """
        Return the list of inner points

        :return: list of points
        :rtype: list<:class:`~numpy.array`>
        """
        return self.points

    def fit(self, points):
        """
        Get a list of points (in numpy.array format) and fit some kind of data structure on them.

        :param points: list of points
        :type points: list<:class:`numpy.array`>
        """
        pass
    
    def k_nearest(self, point, k):
        """
        Given a point, return the k-nearest neighbors to that point.


        :param point: query point
        :type point: :class:`numpy.array`
        :param k: number of neighbors to return (k)
        :type k: int

        :return: k nearest neighbors
        :rtype: list<:class:`numpy.array`
        """
        return []

    def nearest_in_radius(self, point, radius):
        """
        Given a point and a radius, return all the neighbors that have distance <= radius
        
        :param point: query point
        :type point: :class:`~numpy.array`
        :param radius: radius of neighborhood
        :type radius: :class:`numpy.float64`

        :return: nearest neighbors in radius
        :rtype: list<:class:`numpy.array`>
        """
        return []


class NearestNeighborsCached(object):
    """
    Wrapper for a nearest neighbor object that also uses cache,
    and allows for insertion of points in real time.
    This allows adding points lazily, and only when cache is full
    then rebuilding the underlying data structure.

    Example:
        >>> nn = NearestNeighborsCached(NearestNeighbors_sklearn(metric=Metric_Euclidean))
        >>> nn.add_point(Point_d(8, [...]))

    :param nn: nearest neigbors object to wrap
    :type nn: :class:`NearestNeighbors`
    :param max_cache: maximum cache size
    """
    def __init__(self, nn, max_cache=100):
        self.nn = nn
        self.cache = []
        self.max_cache = max_cache

    def get_points(self):
        """
        Return the list of inner points

        :return: list of points
        :rtype: list<:class:`numpy.array`>
        """
        return self.nn.points + self.cache

    def fit(self, points):
        """
        Get a list of points (in CGAL Point_2 or Point_d format) and fit some kind of data structure on them.

        :param points: list of points
        :type points: list<:class:`numpy.array`>
        """
        if len(points) == 0:
            return
        points = points + self.cache # also push anything in cache
        self.nn.fit(points)
        self.cache.clear()

    def add_point(self, point):
        """
        Add a point to the nearest neighbor search

        :param point: point to add
        :type point: :class:`numpy.array`
        """
        self.cache.append(point)
        if len(self.cache) == self.max_cache:
            # if cache is full rebuild nn
            points = self.nn.points + self.cache
            self.nn.fit(points)
            self.cache.clear()
    
    def k_nearest(self, point, k):
        """
        Given a point, return the k-nearest neighbors to that point.

        :param point: query point
        :type point: :class:`numpy.array`
        :param k: number of neighbors to return (k)
        :type k: int

        :return: k nearest neighbors
        :rtype: list<:class:`numpy.array`
        """
        res = self.nn.k_nearest(point, k)
        res += self.cache
        res = sorted(res, key=lambda q: self.nn.metric.dist(point, q).to_double())
        return res[:k]

    def nearest_in_radius(self, point, radius):
        """
        Given a point and a radius, return all the neighbors that have distance <= radius
        
        :param point: query point
        :type point: :class:`numpy.array`
        :param radius: radius of neighborhood
        :type radius: :class:`~numpy.float64`

        :return: nearest neighbors in radius
        :rtype: list<:class:`~numpy.array`
        """
        res = self.nn.nearest_in_radius(point, radius)
        cache_res = []
        for cache_point in self.cache:
            if self.nn.metric.dist(cache_point, point) < radius:
                cache_res.append(cache_point)
        return res + cache_res


class NearestNeighbors_sklearn(NearestNeighbors):
    """
    Sklearn implementation of nearest neighbors

    :param metric: metric to compute nearest neighbors
    :type metric: :class:`solvers.metrics.Metric`
    """
    def __init__(self, metric=Metric_Euclidean):
        super().__init__(metric)
        self.kdtree = None
        self.np_points = []
        self.points = []

    def fit(self, points):
        """
        Get a list of points (in numpy.array format) and fit some kind of data structure on them.

        :param points: list of points
        :type points: list<:class:`~numpy.array`
        """
        if len(points) == 0:
            return

        # Convert points to numpy array
        self.points = points
        self.np_points = np.zeros((len(self.points), 6))
        for i, point in enumerate(self.points):
            self.np_points[i, 0] = point[0]
            self.np_points[i, 1] = point[1]
            self.np_points[i, 2] = point[2]
            self.np_points[i, 3] = point[3]
            self.np_points[i, 4] = point[4]
            self.np_points[i, 5] = point[5]
        self.kdtree = sklearn.neighbors.KDTree(self.np_points, metric=self.metric.sklearn_impl())
    
    def k_nearest(self, point, k):
        """
        Given a point, return the k-nearest neighbors to that point.


        :param point: query point
        :type point: :class:`~numpy.array`
        :param k: number of neighbors to return (k)
        :type k: int

        :return: k nearest neighbors
        :rtype: list<:class:`~numpy.array`>
        """
        if self.kdtree is None:
            return []
        d = point.size
        np_point = np.zeros((1, d))
        for i in range(d):
            np_point[0, i] = point[i]
        _, indices = self.kdtree.query(np_point, k=k)
        res = []
        for idx in indices[0]:
            res.append(self.points[idx])
        return res

    def nearest_in_radius(self, point, radius):
        """
        Given a point and a radius, return all the neighbors that have distance <= radius
        
        :param point: query point
        :type point: :class:`~numpy.array`
        :param radius: radius of neighborhood
        :type radius: :class:`~numpy.float64`

        :return: nearest neighbors in radius
        :rtype: list<:class:`~numpy.array`>
        """
        if self.kdtree is None:
            return []
        d = point.dimension()
        np_point = np.zeros((1, d))
        for i in range(d):
            np_point[0, i] = point[i]
        indices = self.kdtree.query_radius(np_point, r=radius.to_double())
        res = []
        for idx in indices[0]:
            res.append(self.points[idx])
        return res


class NearestNeighbors_CGAL(NearestNeighbors):
    """
    CGAL implementation of nearest neighbors

    :param metric: metric to compute nearest neighbors
    :type metric: :class:`metrics.Metric`
    """
    def __init__(self, metric=Metric_Euclidean):
        super().__init__(metric)
        self.kdtree = None
        self.points = []
        self.points_d = []

    def fit(self, points):
        """
        Get a list of points (in numpy.array format) and fit some kind of data structure on them.

        :param points: list of points
        :type points: list<:class:`numpy.array`>
        """
        if len(points) == 0:
            return

        # Convert points to numpy array
        self.points = points
        self.points_d = points
        self.kdtree = Ss.Kd_tree(self.points_d)
    
    def _convert_results_to_points(self, res):
        """
        Get array of results and convert it to original points
        (i.e. if points were Point_2 convert Point_d to Point_2)
        """
        return res

    def k_nearest(self, point, k):
        """
        Given a point, return the k-nearest neighbors to that point.


        :param point: query point
        :type point: :class:`numpy.array`
        :param k: number of neighbors to return (k)
        :type k: int

        :return: k nearest neighbors
        :rtype: list<:class:`numpy.array`
        """
        if self.kdtree is None:
            return []
        query_d = point
        eps = 0.0
        search_nearest = True
        sort_neighbors = True
        distance = self.metric.sklearn_impl()
        
        search = Ss.K_neighbor_search(
            self.kdtree, query_d, k, eps, 
            search_nearest, distance, sort_neighbors)
        lst = []
        search.k_neighbors(lst)
        res = [p[0] for p in lst]
        return self._convert_results_to_points(res)

    def nearest_in_radius(self, point, radius):
        """
        Given a point and a radius, return all the neighbors that have distance <= radius
        
        :param point: query point
        :type point: :class:`~numpy.array`
        :param radius: radius of neighborhood
        :type radius: :class:`~numpy.float64`

        :return: nearest neighbors in radius
        :rtype: list<:class:`~numpy.array`>
        """
        if self.kdtree is None:
            return []
        query_d = point
        eps = 0.0
        sphere = Ss.Fuzzy_sphere(query_d, radius, eps)
        res = []
        self.kdtree.search(sphere, res)
        return self._convert_results_to_points(res)
