import trimesh
from geometry_utils.collision_detection import ObjectCollisionDetection
from trimesh.collision import CollisionManager
from scipy.spatial.transform import Rotation as R

from solvers import RobotPolygon

EDGE_SAMPLES = 1000


class ObjectCollisionDetectionVerify(ObjectCollisionDetection):
    def __init__(self, obstacles, robot, offset=0.05, max_depth=8):
        super().__init__(obstacles, robot, offset)
        self.edge_samples = EDGE_SAMPLES

    def is_point_valid(self, point):
        if isinstance(self.robot, RobotPolygon):
            robot_mesh = self.robot.poly
            translation_vector = point[0:3] - self.robot.start[0:3]
            moved_vertices = robot_mesh.vertices + translation_vector
            rotation_x = R.from_euler('x', point[3])
            rotation_y = R.from_euler('y', point[4])
            rotation_z = R.from_euler('z', point[5])
            rotated_vertices = rotation_z.apply(rotation_y.apply(rotation_x.apply(moved_vertices)))
            new_pos_robot = trimesh.Trimesh(vertices=rotated_vertices, faces=robot_mesh.faces)
            # Check for collisions
            for obstacle in self.obstacles:
                collision_manager = CollisionManager()
                collision_manager.add_object('new_pos_robot', new_pos_robot)
                collision_manager.add_object('obstacle', obstacle.poly)
                if collision_manager.in_collision_internal():
                    return False
        return True
