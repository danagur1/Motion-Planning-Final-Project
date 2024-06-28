import trimesh
import numpy as np
import time

from geometry_utils.bounding_boxes import calc_scene_bounding_box
from solvers import Scene

EPS = 0.001
EDGE_SAMPLES_DEFAULT = 10


class ObjectCollisionDetection(object):
    """
    A class object that handles collision detection of a single object with obstacles.
    The collision detector builds a CGAL arrangement representing the scene and allows to
    (quickly) query the arrangement for collisions.

    :param obstacles: list of obstacles
    :type obstacles: list<:class:`~solvers.Obstacle`>
    :param robot: robot for building the collision detection
    :type robot: :class:`~solvers.Robot`
    :param offset: offset for rod edge collision detection
    :type offset: :class:`~numpy.float64`
    """

    def __init__(self, obstacles, robot, offset=0.05):
        self.obstacles = obstacles
        self.robot = robot
        self.cspace = None
        self.point_location = None
        self.edge_samples = EDGE_SAMPLES_DEFAULT  # the amount of samples to check on edge
        self.bounding_boxes = self.set_bounding_boxes()
        self.sdf_functions = [(lambda point: self.sdf(obstacle_idx, point)) for obstacle_idx in range(len(obstacles))]

    def is_edge_valid(self, edge):
        """
        Check if a edge (start point with angle to end point with angle) is valid (i.e. not colliding with anything).

        :param edge: edge to check
        :type edge: :class:`~discopygal.bindings.Segment_2`

        :return: False if edge intersects with the interior of an obstacle
        :rtype: :class:`bool`
        """
        # TO DO: handle ball robot
        edge_point1 = edge[0]
        edge_point2 = edge[1]
        for epsilon in np.arange(0, 1, 1 / self.edge_samples):
            check_point = edge_point1 * epsilon + edge_point2 * (1 - epsilon)
            if not self.is_point_valid(check_point):
                return False
            """if count_checks % 10 == 0:
                print("checked " + str(count_checks))
            count_checks += 1"""
        return True

    def sdf(self, obstacle_idx, point):
        # should be overridden
        return np.inf

    def set_bounding_boxes(self):
        return [calc_scene_bounding_box(Scene(obstacles=[self.obstacles[obstacle_idx]]))
                for obstacle_idx in range(len(self.obstacles))]

    def is_point_valid(self, point):
        """
        Check if a point is valid (i.e. not colliding with anything).

        :param point: point to check
        :type point: :class:`~trimesh.caching.TrackedArray`

        :return: False if point lies in the interior of an obstacle
        :rtype: :class:`bool`
        """
        for sample in self.robot.poly.sample(self.edge_samples):
            sample = np.concatenate([np.array(sample), np.array([0, 0, 0])]) + np.array(point)
            new_sample = self.rotate_point(self.rotate_point(self.rotate_point(sample[:3], sample[3], 'x'),
                                                             sample[4], 'y'), sample[5], 'z')
            for obstacle_idx in range(len(self.obstacles)):
                if self.sdf_functions[obstacle_idx](new_sample) < 0:
                    return False
        return True

    def rotate_point(self, point, angle, axis):
        if axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif axis == 'z':
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        rotated_point = np.dot(R, point)
        return rotated_point

    def sdf_for_sample_point(self, mesh, point):
        closest_point, distance, _ = mesh.nearest.on_surface([point])
        inside = mesh.contains([point])[0]
        signed_distance = distance if not inside else -distance
        return signed_distance
