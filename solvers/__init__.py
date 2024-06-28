from PyQt5 import QtCore
import numpy as np

from geometry_utils import conversions


def load_object_from_dict(d):
    """
    Load a seriallized object from a dict
    In order for this to work, the dict should have a "__class__" property,
    which is equal to the exact python name of the class

    :param d: dict describing an object
    :type d: :class:`dict`

    :return: the serialized object
    :rtype: :class:`object`
    """
    klass = globals()[d['__class__']]
    obj = klass.from_dict(d)
    return obj


class Robot(object):
    """
    Abstact class that represents the notion of a robot in a scene.
    Reference point is always the origin of the given geometry.

    :param start: The start location of the robot (unless stated otherwise)
    :type start: :class:`numpy.array`
    :param end: The end location of the robot (unless stated otherwise)
    :type end: :class:`numpy.array`
    :param data: Any metadata appended to the robot (could be None)
    :type data: :class:`object`
    """
    def __init__(self, start, end, data=None):
        if len(start) == 3:
            self.start = np.concatenate((start, [0, 0, 0]), axis=0)
        else:
            self.start = start
        if len(end) == 3:
            self.end = np.concatenate((end, [0, 0, 0]), axis=0)
        else:
            self.end = end
        self.data = data or {}

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'Robot',
            'start': conversions.Point_3_to_xyz(self.start),
            'end': conversions.Point_3_to_xyz(self.end),
            'data': self.data
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`
        """
        return Robot(
            start=conversions.xy_to_Point_2(*d['start']),
            end = conversions.xy_to_Point_2(*d['end']),
            data = d['data']
        )


class RobotBall(Robot):
    """
    A disc robot. Its geometry is defined by its radius.

    :param radius: The radius of the disc robot, as a CGAL field type
    :type radius: :class:`numpy.float64`
    :param start: The start location of the robot, as a CGAL point
    :type start: :class:`numpy.array`
    :param end: The end location of the robot, as a CGAL point
    :type end: :class:`numpy.array`
    :param data: Any metadata appended to the robot (could be None)
    :type data: :class:`object`
    """
    def __init__(self, radius, start, end, data=None):
        super().__init__(start, end, data)
        self.radius = radius

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'RobotBall',
            'radius': conversions.FT_to_float(self.radius),
            'start': conversions.Point_2_to_xy(self.start),
            'end': conversions.Point_2_to_xy(self.end),
            'data': self.data
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`
        """
        return RobotBall(
            radius = conversions.float_to_FT(d['radius']),
            start = conversions.xy_to_Point_2(*d['start']),
            end = conversions.xy_to_Point_2(*d['end']),
            data = d['data']
        )


class RobotPolygon(Robot):
    # TO DO: change polygon names to polyheders
    """
    A polygonal robot. Its geometry is given as a CGAL 2D polygon.

    :param poly: The geometry of the robot, as a CGAL 2D Polygon
    :type poly: :class:`numpy.array`
    :param start: The start location of the robot, as a CGAL point
    :type start: :class:`numpy.array`
    :param end: The end location of the robot, as a CGAL point
    :type end: :class:`numpy.array`
    :param data: Any metadata appended to the robot (could be None)
    :type data: :class:`object`
    """
    def __init__(self, poly, start, end, data=None):
        super().__init__(start, end, data)
        self.poly = poly

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'RobotPolygon',
            'poly': conversions.Polygon_2_to_array_of_points(self.poly),
            'start': conversions.Point_2_to_xy(self.start),
            'end': conversions.Point_2_to_xy(self.end),
            'data': self.data
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`
        """
        return RobotPolygon(
            poly = conversions.array_of_points_to_Polygon_2(d['poly']),
            start = conversions.xy_to_Point_2(*d['start']),
            end = conversions.xy_to_Point_2(*d['end']),
            data = d['data']
        )


class RobotRod(Robot):
    """
    A rod robot. Its geometry is defined by its length.

    :param length: The length of the rod, as a CGAL field type
    :type length: :class:`numpy.float64`
    :param start: The start location and angle of the robot, as a tuple of CGAL point and angle
    :type start: (:class:`numpy.array`, :class:`numpy.float64`)
    :param end: The end location and angle of the robot, as a tuple of CGAL point and angle
    :type end: (:class:`numpy.array`, :class:`numpy.float64`)
    :param data: Any metadata appended to the robot (could be None)
    :type data: :class:`object`
    """
    def __init__(self, length, start, end, data=None):
        super().__init__(start, end, data)
        self.length = length

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'RobotRod',
            'length': conversions.FT_to_float(self.length),
            'start': (conversions.Point_2_to_xy(self.start[0]), conversions.FT_to_float(self.start[1])),
            'end': (conversions.Point_2_to_xy(self.end[0]), conversions.FT_to_float(self.end[1])),
            'data': self.data
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`
        """
        return RobotRod(
            length = conversions.float_to_FT(d['length']),
            start = (conversions.xy_to_Point_2(*d['start'][0]), conversions.float_to_FT(d['start'][1])),
            end = (conversions.xy_to_Point_2(*d['end'][0]), conversions.float_to_FT(d['end'][1])),
            data = d['data']
        )


class Obstacle(object):
    """
    Abstract class that represents the notion of an obstacle in the scene.
    The obstacle has some geometry.

    :param data: Any metadata appended to the obstacle (could be None)
    :type data: :class:`object`
    """
    def __init__(self, data):
        self.data = data or {}

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'Obstacle',
            'data': self.data
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`
        """
        return Obstacle(
            data = d['data']
        )


class ObstacleDisc(Obstacle):
    """
    Disc obstacle in the scene. Its geometry is given as a location and radius.

    :param location: Disc obstacle location point, as a CGAL point
    :type location: :class:`numpy.array`
    :param radius: Disc radius, as a CGAL field type
    :type radius: :class:`numpy.float64`
    """
    def __init__(self, location, radius, data=None):
        super().__init__(data)
        self.location = location
        self.radius = radius

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'ObstacleDisc',
            'location': conversions.Point_2_to_xy(self.location),
            'radius': conversions.FT_to_float(self.radius),
            'data': self.data
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`
        """
        return ObstacleDisc(
            location = conversions.xy_to_Point_2(*d['location']),
            radius = conversions.float_to_FT(d['radius']),
            data = d['data']
        )


class ObstaclePolygon(Obstacle):
    """
    Polygon obstacle in the scene. Its geometry is given as a polygon.

    :param poly: Polygon obstacle geometry, as a CGAL polygon
    :type poly: :class:`numpy.array`
    """
    def __init__(self, poly, data=None):
        super().__init__(data)
        self.poly = poly

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'ObstaclePolygon',
            'poly': conversions.Polyhedron_3_to_array_of_points(self.poly),
            'data': self.data
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`
        """
        return ObstaclePolygon(
            poly = conversions.array_of_points_to_Polygon_2(d['poly']),
            data = d['data']
        )


class Scene(object):
    """
    The notion of "scene" in DiscoPygal, which is the setting where we conduct motion planning.
    A scene has robots that can move inside it, and obstacles.
    Also the scene can have any metadata, saved as a dictionary.

    :param obstacles: list of obstacles
    :type obstacles: list<:class:`Obstacle`>
    :param robots: list of robots
    :type robots: list<:class:`Robot`>
    :param metadata: dict with metadata on the scene
    :type metadata: :class:`dict`
    """
    def __init__(self, obstacles=None, robots=None, metadata=None):
        self.obstacles = obstacles or []
        self.robots = robots or []
        self.metadata = metadata or {}

    def clear(self):
        self.robots.clear()
        self.obstacles.clear()

    def add_robot(self, robot):
        """
        Add a robot to the scene

        :param robot: Robot to add
        :type robot: :class:`Robot`
        """
        if robot not in self.robots:
            self.robots.append(robot)

    def remove_robot(self, robot):
        """
        Remove a robot from the scene

        :param robot: Robot to remove
        :type robot: :class:`Robot`
        """
        if robot in self.robots:
            self.robots.remove(robot)

    def add_obstacle(self, obstacle):
        """
        Add a obstacle to the scene

        :param obstacle: obstacle to add
        :type obstacle: :class:`Obstacle`
        """
        if obstacle not in self.obstacles:
            self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle):
        """
        Remove a obstacle from the scene

        :param obstacle: obstacle to remove
        :type obstacle: :class:`Obstacle`
        """
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)

    def to_dict(self):
        """
        Convert current object to json dict

        :return: dict representing json export
        :rtype: :class:`dict`
        """
        return {
            '__class__': 'Scene',
            'obstacles': [obstacle.to_dict() for obstacle in self.obstacles],
            'robots': [robot.to_dict() for robot in self.robots],
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(d):
        """
        Load json dict to object

        :param d: dict representing json export
        :type d: :class:`dict`

        :return: A scene object build from the given dict
        :rtype: :class:`Scene`
        """
        return Scene(
            obstacles=[load_object_from_dict(obstacle) for obstacle in d['obstacles']],
            robots=[load_object_from_dict(robot) for robot in d['robots']],
            metadata=d['metadata']
        )

    def calc_max_robot_size(self):
        """
        Return the size of the largest robot in the scene
        For RobotBall the is it's diameter
        For RobotRod the size is it's length
        """
        robots_sizes = []
        for robot in self.robots:
            if type(robot) is RobotBall:
                robots_sizes.append(robot.radius * 2)
            elif type(robot) is RobotRod:
                robots_sizes.append(robot.length)
            elif type(robot) is RobotPolygon:
                # TODO: calc also size for RobotPolygon
                return 0

        return max(robots_sizes)


class PathPoint(object):
    """
    A single point in the path (of some robot).
    Has a 3D location and additional data

    :param location: location of point
    :type location: :class:`numpy.array`
    :param data: attached data to point
    :type data: :class:`dict`
    """
    def __init__(self, location, data=None):
        self.location = location
        self.data = data or {}

class Path(object):
    """
    Representation of the path a single robot does

    :param points: points along the path
    :type points: list<:class:`PathPoint`>
    """
    def __init__(self, points):
        self.points = points


class PathCollection(object):
    """
    Collection of the paths of all the robots in the scene.
    This is the objects that is returned by a solver.

    :param paths: collection of paths
    :type paths: dict<:class:`Robot`, :class:`Path`>
    """
    def __init__(self, paths=None):
        self.paths = paths or {}

    def add_robot_path(self, robot, path):
        """
        Add a robot's path to the collection

        :param robot: the robot we add
        :type robot: :class:`Robot`
        :param path: robot's path
        :type path: :class:`Path`
        """
        self.paths[robot] = path


