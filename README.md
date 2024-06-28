# Algorithmic Robotics And Motion Planning Project
This project in Algorithmic Robotics And Motion Planning explores efficient motion planning for robots navigating in 3D spaces containing polyhedral obstacles. The key focus is on implementing various methods for computing the Signed Distance Function (SDF) for collision detection in probabilistic motion planning algorithms.
The implementation is based on custom modifications of the discopygal Python package, specifically adapted for 3D computations.

## Requierments
Install following python packages: trimesh, PyQt5, rtree, python-fcl

## How to run the project?
'main' file contain all the tests for all the samples

You can use any scene file in .obj format. Every robot name must start with "robot" and every obstacle name must start 
with "obstacle". Some example scenes are in the scenes directory.
The parse_file function (in main) will return a scene that matches the input file
Then create PRM object and call load_scene and solve

### Running Example
```python
scene = parse_file(r"../scenes/scene6.obj")
solver = PRM(num_landmarks=100, k=15, cd_method=CD_METHOD_NN, num_points=num_points)
solver.load_scene(scene)
path_collection = solver.solve()
```
