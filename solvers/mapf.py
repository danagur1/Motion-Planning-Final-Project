from . import Robot
from . import PathPoint, Path, PathCollection
from .Solver import Solver

class GridMAPFSolver(Solver):
    """
    Abstract solver wrapper for grid Multi-Agent Path Finding (MAPF) problems.
    Inherit this class to use a more MAPF-oriented interface (instead of the
    classis DiscoPygal), while still being able to use the designated scene designer
    and solver viewer.
    """
    def load_scene(self, scene):
        super().load_scene(scene)
        
        self.robots = []
        for robot in self.scene.robots:
            if type(robot) is not RobotDisc:
                continue