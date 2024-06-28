from collections import namedtuple

from .transform import offset_polygon
from solvers import Scene, RobotBall, RobotPolygon, ObstacleDisc, ObstaclePolygon

BoundingBox = namedtuple("BoundingBox", "min_x max_x min_y max_y min_z max_z")


def calc_scene_bounding_box(scene: Scene):
    """
    Get a DiscoPygal scene and compute its bounding box.
    The bounding box is computed as the smallest axis-aligned box
    that contains all the obstacles and robots.

    :param scene: scene
    :type scene: :class:`~solvers.Scene`

    :return: min_x, max_x, min_y, max_y [bounds of the scene]
    :rtype: (:class:`~numpy.float64`, :class:`~numpy.float64`, :class:`~numpy.float64`, :class:`~numpy.float64`)
    """
    X = []
    Y = []
    Z = []

    for obstacle in scene.obstacles:
        if type(obstacle) is ObstaclePolygon:
            for point in obstacle.poly.vertices:
                X.append(point[0])
                Y.append(point[1])
                Z.append(point[2])
        elif type(obstacle) is ObstacleDisc:
            X.append(obstacle.location[0] - obstacle.radius)
            X.append(obstacle.location[0] + obstacle.radius)
            Y.append(obstacle.location[1] - obstacle.radius)
            Y.append(obstacle.location[1] + obstacle.radius)
            Z.append(obstacle.location[1] - obstacle.radius)
            Z.append(obstacle.location[1] + obstacle.radius)

    for robot in scene.robots:
        if type(robot) is RobotPolygon:
            poly1 = offset_polygon(robot.poly, robot.start)
            poly2 = offset_polygon(robot.poly, robot.end)
            for point in poly1:
                X.append(point[0])
                Y.append(point[1])
                Z.append(point[2])
            for point in poly2:
                X.append(point[0])
                Y.append(point[1])
                Z.append(point[2])
        elif type(robot) is RobotBall:
            X.append(robot.start[0] - robot.radius)
            X.append(robot.start[0] + robot.radius)
            Y.append(robot.start[1] - robot.radius)
            Y.append(robot.start[1] + robot.radius)
            Z.append(robot.start[2] - robot.radius)
            Z.append(robot.start[2] + robot.radius)
            X.append(robot.end[0] - robot.radius)
            X.append(robot.end[0] + robot.radius)
            Y.append(robot.end[1] - robot.radius)
            Y.append(robot.end[1] + robot.radius)
            Z.append(robot.end[2] - robot.radius)
            Z.append(robot.end[2] + robot.radius)

    min_x = min(X)
    max_x = max(X)
    min_y = min(Y)
    max_y = max(Y)
    min_z = min(Z)
    max_z = max(Z)

    return BoundingBox(min_x, max_x, min_y, max_y, min_z, max_z)
