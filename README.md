**Requierments**
Install following python packages: trimesh, PyQt5, rtree, python-fcl

**How to run the project?**
the file main contain all the tests for all the samples

You can use any scene file in .obj format. Every robot name must start with "robot" and every obstacle name must start 
with "obstacle". Some example scenes are in the scenes directory.
The parse_file function (in main) will return a scene that matches the input file
Then create PRM object and call load_scene and solve

For example:
scene = parse_file(r"../scenes/scene6.obj")
solver = PRM(num_landmarks=100, k=15, cd_method=CD_METHOD_NN, num_points=num_points)
if not solver:
    print("time exceeded")
solver.load_scene(scene)
path_collection = solver.solve() 