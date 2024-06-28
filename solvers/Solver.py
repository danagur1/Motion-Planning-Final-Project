import sys
from abc import abstractmethod
import networkx as nx
import numpy as np

from geometry_utils.bounding_boxes import calc_scene_bounding_box, BoundingBox
from solvers import Scene, PathPoint, PathCollection, Path, RobotRod


class Solver(object):
    """
    The main solver class. Every implemented solver should derive from it.

    :param bounding_margin_width_factor:
                                         | The factor which by to increase the solver's bounding box margin
                                         | Set 0 for a tight bounding box
                                         | Set -1 for no bounding box (default)

    :type bounding_margin_width_factor: :class:`~numpy.float64`

    :param scene: The loaded scene object the solver will solve
    :type scene: :class:`~solvers.Scene`

    :param verbose: Should solver print logs during solving process
    :type verbose: :class:`bool`
    """
    DEFAULT_BOUNDS_MARGIN_FACTOR = 2
    NO_BOUNDING_BOX = -1

    def __init__(self, bounding_margin_width_factor=NO_BOUNDING_BOX):
        self.scene = None
        self.writer = None
        self.verbose = False
        self.bounding_margin_width_factor = bounding_margin_width_factor
        self._bounding_box = None

    @classmethod
    def init_default_solver(cls):
        """
        Create a solver with default parameters of given solver class
        """
        default_args = {}
        for arg, (_, default_value, arg_type) in cls.get_arguments().items():
            default_args[arg] = arg_type(default_value)
        return cls(**default_args)


    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Derived solvers can override this method to also add
        some sort of pre-processing to the scene

        :param scene: scene to load
        :type scene: :class:`solvers.Scene`
        """
        self.scene = scene
        self._bounding_box = self.calc_bounding_box()

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        The base solver returns for each robot a simple path of its start and end position -
        which for most scenes might not be valid!

        :return: path collection of motion planning
        :rtype: :class:`solvers.PathCollection`
        """
        path_collection = PathCollection()
        for robot in self.scene.robots:
            if type(robot) is RobotRod:
                start_location = robot.start[0]
                start_data = {'angle': robot.start[1]}
                end_location = robot.end[0]
                end_data = {'angle': robot.end[1]}
            else:
                start_location = robot.start
                start_data = {}
                end_location = robot.end
                end_data = {}
            start_point = PathPoint(start_location, start_data)
            end_point = PathPoint(end_location, end_data)
            path = Path([start_point, end_point])
            path_collection.add_robot_path(robot, path)
        return path_collection

    def set_verbose(self, writer=None):
        """
        Call this method to set a verbose solver, i.e. print while solving.
        """
        self.verbose = True
        self.writer = writer
        if self.writer is None:
            self.writer = sys.stdout

    def disable_verbose(self):
        """
        Call this method to disable verbose running for the solver
        """
        self.writer = None
        self.verbose = False

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridded by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or None
        """
        return None

    def get_arrangement(self):
        """
        Return an arrangement (if applicable).
        Can be overridded by solvers.

        :return: arrengement
        :rtype: :class:`List<trimesh.base.Trimesh>`
        """
        return None

    def get_bounding_box_graph(self):
        """
        Return the graph of the bounding box the solver calculated for the loaded scene (if used one)

        :return: bounding_box_graph
        :rtype: :class:`networkx.Graph` or None
        """
        if self._bounding_box is None:
            return None

        bounding_box_graph = nx.Graph()
        left_bottom = np.array([self._bounding_box.min_x, self._bounding_box.min_y])
        left_top = np.array([self._bounding_box.min_x, self._bounding_box.max_y])
        right_bottom = np.array([self._bounding_box.max_x, self._bounding_box.min_y])
        right_top = np.array([self._bounding_box.max_x, self._bounding_box.max_y])
        for point in [left_bottom, left_top, right_bottom, right_top]:
            bounding_box_graph.add_node(point)

        bounding_box_graph.add_edge(left_bottom, left_top)
        bounding_box_graph.add_edge(left_top, right_top)
        bounding_box_graph.add_edge(right_top, right_bottom)
        bounding_box_graph.add_edge(right_bottom, left_bottom)
        return bounding_box_graph

    @staticmethod
    @abstractmethod
    def get_arguments():
        """
        Return a list of arguments and their description, defaults and types.
        Can be used by a GUI to generate fields dynamically.
        Should be overridded by solvers.

        :return: arguments dict
        :rtype: :class:`dict`
        """
        raise NotImplementedError()

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridded by solvers.

        :param d: arguments dict
        :type d: :class:`dict`

        :return: solver object with arguments as in dict d
        :rtype: :class:`Solver`

        .. deprecated:: 1.0.3
            Use :func:`init_default_solver` instead
        """
        return Solver()

    def update_arguments(self, args: dict):
        """
        Update the arguments solver

        :param args: arguments dict
        :type args: :class:`dict`
        """
        self.__dict__.update(args)

    def log(self, text, **kwargs):
        """
        | Print a log to screen and to gui is enabled
        | Prints only if set to be verbose (automatically done in solver_viewer)

        :param text: text to print
        :param kwargs: more key-word arguments to pass to :func:`print` builtin function
        """
        if self.verbose:
            print(text, file=self.writer, **kwargs)

    def calc_bounding_box(self):
        if self.scene is None or self.bounding_margin_width_factor < 0:
            return None

        margin_width = self.scene.calc_max_robot_size() * self.bounding_margin_width_factor
        tight_bounding_box = calc_scene_bounding_box(self.scene)
        return BoundingBox(tight_bounding_box.min_x - margin_width,
                           tight_bounding_box.max_x + margin_width,
                           tight_bounding_box.min_y - margin_width,
                           tight_bounding_box.max_y + margin_width,
                           tight_bounding_box.min_z - margin_width,
                           tight_bounding_box.max_z + margin_width,
                           )
