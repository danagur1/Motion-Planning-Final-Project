import numpy as np
import trimesh

from solvers import Scene, RobotBall, ObstaclePolygon, RobotPolygon
from solvers.prm import PRM, CD_METHOD_GRID, CD_METHOD_HIERARCHICAL, CD_METHOD_NN
from solvers.verify_paths import verify_paths

FILE_PATH = r"../scenes/scene2.obj"


def parse_file(file_path):
    scene = Scene()
    file_data = trimesh.load(file_path)
    for name, mesh in file_data.geometry.items():
        if name.startswith("obstacle"):
            scene.add_obstacle(ObstaclePolygon(mesh))
        else:  # a robot
            scene.add_robot(RobotPolygon(mesh, np.array([0, 0, 3]), np.array([0, 0, -4])))
    return scene


def solve_by_nn(scene):
    for num_points in [1000, 10000]:
        print("solving using NN method with num_points="+str(num_points))
        # "Solve" the scene (find paths for the robots)
        solver = PRM(num_landmarks=100, k=15, cd_method=CD_METHOD_NN, num_points=num_points)
        if not solver:
            print("time exceeded")
        solver.load_scene(scene)
        path_collection = solver.solve()  # Returns a PathCollection object

        # Print the points of the paths
        """for i, (robot, path) in enumerate(path_collection.paths.items()):
            print("Path for robot {}:".format(i))
            for point in path.points:
                print('\t', point.location)  # point is of type PathPoint, point.location is CGALPY.Ker.Point_2"""

        result, reason = verify_paths(scene, path_collection)
        print("result="+str(result))
        # print(f"Are paths valid: {result}")


def solve_by_grid(scene):
    for edge_samples in [10, 50]:
        for grid_samples in [3, 4, 5]:
            for grid_resolution in [3, 4, 5]:
                print("solving using grid method with edge_samples=" + str(edge_samples)+" grid_samples=" + str(grid_samples) +
                      " grid_resolution" + str(grid_resolution))
                solver = PRM(num_landmarks=100, k=15, cd_method=CD_METHOD_NN, edge_samples=edge_samples,
                             grid_samples=grid_samples, grid_resolution=grid_resolution)
                if not solver:
                    print("time exceeded")
                solver.load_scene(scene)
                path_collection = solver.solve()  # Returns a PathCollection object
        result, reason = verify_paths(scene, path_collection)
        print("result="+str(result))


def solve_by_hierarchical(scene):
    for max_depth in [2, 3, 4]:
        for edge_samples in [10, 20, 100]:
            for bbox_samples in [5, 7, 10]:
                print("solving using hierarchical method with max_depth=" + str(max_depth)+" edge_samples=" + str(edge_samples) +
                      " bbox_samples" + str(bbox_samples))
                solver = PRM(num_landmarks=100, k=15, cd_method=CD_METHOD_NN, max_depth=max_depth, edge_samples=edge_samples,
                             bbox_samples=bbox_samples)
                if not solver:
                    print("time exceeded")
                solver.load_scene(scene)
                path_collection = solver.solve()  # Returns a PathCollection object
        result, reason = verify_paths(scene, path_collection)
        print("result="+str(result))


for i in range(5, 15):
    scene = parse_file(r"../scenes/scene"+str(i)+".obj")
    solve_by_grid(scene)
    solve_by_hierarchical(scene)
    solve_by_nn(scene)
