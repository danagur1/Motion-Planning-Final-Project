import numpy as np

from . import *
from geometry_utils.verify import  ObjectCollisionDetectionVerify

def verify_paths(scene: Scene, paths: PathCollection):
    """
    Get a scene and a path, and verify that the given path collection is indeed a valid solution

    :param scene: scene
    :type scene: :class:`solvers.Scene`
    :param paths: paths to verify
    :type paths: :class:solvers.PathCollection`

    :return: True if paths are valid and if invalid, return a reason.
    :rtype: :class:`bool`, :class:`str`
    """
    if paths is None:
        return False, "Paths is None"

    # Check collision of each robot against obstacles
    lengths = []
    for robot in paths.paths:
        path = paths.paths[robot]
        lengths.append(len(path.points))
        cd = ObjectCollisionDetectionVerify(scene.obstacles, robot)
        for i in range(len(path.points) - 1):
            edge = np.array([path.points[i].location, path.points[i + 1].location])
            if not cd.is_edge_valid(edge):
                return False, f"Collision with obstacle. Edge: {i}"

    # Check that all paths are of same length
    if len(lengths) == 0:
        return False, "Empty path"
    if min(lengths) != max(lengths):
        return False, "Paths are not of the same length"
    path_len = max(lengths)

    # Check collision of any two robots
    robots = list(paths.paths.keys())
    for k in range(path_len-1):
        for i in range(len(robots)):
            for j in range(i+1, len(robots)):
                robot1 = robots[i]
                robot2 = robots[j]
                edge1 = np.array([paths.paths[robot1].points[k].location, paths.paths[robot1].points[k+1].location])
                edge2 = np.array([paths.paths[robot2].points[k].location, paths.paths[robot2].points[k+1].location])
                if collision_detection.collide_two_robots(robot1, edge1, robot2, edge2):
                    return False, "Collision with robots"
    
    # Check that start and end location is valid
    for robot in robots:
        if type(robot) is RobotRod:
            start = (paths.paths[robot].points[0].location, paths.paths[robot].points[0].data['theta'])
            end = (paths.paths[robot].points[-1].location, paths.paths[robot].points[-1].data['theta'])
        else:
            start = paths.paths[robot].points[0].location
            end = paths.paths[robot].points[-1].location
        if (list(robot.start) != start) or (list(robot.end) != end):
            return False, "Robots start/end goal mismatch"
    
    return True, ""


