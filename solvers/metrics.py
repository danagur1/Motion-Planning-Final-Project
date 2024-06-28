import math

import numpy as np

import sklearn.metrics

class MetricNotImplemented(Exception):
    pass


class Metric(object):
    """
    Representation of a metric for nearest neighbor search.
    Should support all kernels/methods for nearest neighbors
    (like CGAL and sklearn).
    """ 
    def __init__(self):
        pass
    
    @staticmethod
    def sklearn_impl():
        """
        Return the metric as sklearn metric object 
        """
        raise MetricNotImplemented('sklearn')


class Metric_Euclidean(Metric):
    """
    Implementation of the Euclidean metric for nearest neighbors search
    """
    @staticmethod
    def dist(p, q):
        """
        Return the distance between two points

        :param p: first point
        :type p: :class:`~np.array`
        :param q: second point
        :type q: :class:`~np.array`

        :return: distance between p and q
        :rtype: :class:np.float64
        """
        squared_diff_sum = sum((a - b) ** 2 for a, b in zip(p, q))
        # Return the square root of the sum
        return math.sqrt(squared_diff_sum)
    
    @staticmethod
    def sklearn_impl():
        """
        Return the metric as sklearn metric object 
        """
        return sklearn.metrics.DistanceMetric.get_metric('euclidean')
