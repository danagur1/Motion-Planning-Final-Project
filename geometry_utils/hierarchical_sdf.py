from collections import namedtuple
from geometry_utils.collision_detection import ObjectCollisionDetection
import numpy as np
import time

# Define a simple Octree node structure
OctreeNode = namedtuple('OctreeNode', ['bbox', 'children', 'sdf_value'])

EDGE_SAMPLES = 10
BBOX_SAMPLES = 5
EQUALITY_EPSILON = 0.000001


class ObjectCollisionDetectionHierarchical(ObjectCollisionDetection):
    def __init__(self, obstacles, robot, offset=0.05, max_depth=3, edge_samples=10, bbox_samples=5):
        super().__init__(obstacles, robot, offset)
        self.edge_samples = edge_samples
        self.bbox_samples = bbox_samples
        self.max_depth = max_depth
        self.roots = [None]*len(self.obstacles)
        self.start_time = time.time()
        self.time_exceeded = False
        for obstacle_idx in range(len(self.obstacles)):
            root_bbox = self.bounding_boxes[obstacle_idx]
            root_bbox_tuple = (root_bbox.min_x, root_bbox.max_x, root_bbox.min_y, root_bbox.max_y, root_bbox.min_z,
                               root_bbox.max_z)
            # print("building tree for obstacle idx " + str(obstacle_idx))
            self.roots[obstacle_idx] = self.build_octree(obstacle_idx, root_bbox_tuple, 0)
            if self.time_exceeded:
                return False

    def build_octree(self, obstacle_idx, bbox, depth):
        if time.time() - self.start_time > 1200:
            self.time_exceeded = True
            return False
        sdf_samples_values = self.sample_sdf(bbox, obstacle_idx)
        bbox_sdf_value = self.compute_sdf(sdf_samples_values)

        if (depth >= self.max_depth) or (self.all_bbox_equal(sdf_samples_values)):
            return OctreeNode(bbox, None, bbox_sdf_value)

        # print(str(obstacle_idx) + str(bbox) + str(max(sdf_samples_values)-min(sdf_samples_values)))

        children_bbox = self.split_bbox(bbox)
        children = [self.build_octree(obstacle_idx, child_bbox, depth + 1) for child_bbox in children_bbox]

        return OctreeNode(bbox, children, bbox_sdf_value)

    def split_bbox(self, bbox):
        min_x, max_x, min_y, max_y, min_z, max_z = bbox
        mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2

        children_bbox = [
            (min_x, mid_x, min_y, mid_y, min_z, mid_z),
            (mid_x, max_x, min_y, mid_y, min_z, mid_z),
            (min_x, mid_x, mid_y, max_y, min_z, mid_z),
            (mid_x, max_x, mid_y, max_y, min_z, mid_z),
            (min_x, mid_x, min_y, mid_y, mid_z, max_z),
            (mid_x, max_x, min_y, mid_y, mid_z, max_z),
            (min_x, mid_x, mid_y, max_y, mid_z, max_z),
            (mid_x, max_x, mid_y, max_y, mid_z, max_z)
        ]

        return children_bbox

    def sample_sdf(self, bbox, obstacle_idx):
        sdf_samples_values = []
        obstacle = self.obstacles[obstacle_idx].poly
        for sample in self.iterate_bbox_points(bbox):
            sdf_samples_values.append(self.sdf_for_sample_point(obstacle, sample))
        return sdf_samples_values  # Use mean as a simple aggregation method

    def compute_sdf(self, sdf_samples_values):
        return np.mean(sdf_samples_values)

    def all_bbox_equal(self, sdf_samples_values):
        first_val = sdf_samples_values[0]
        return all(np.abs(val - first_val) < EQUALITY_EPSILON for val in sdf_samples_values)

    def sdf(self, obstacle_idx, point):
        tree_root = self.roots[obstacle_idx]
        if not self.point_in_bbox(tree_root.bbox, point):
            return np.inf
        return self.sdf_recursive(point, tree_root)

    def sdf_recursive(self, point, node):
        if not node.children:
            return node.sdf_value

        for idx, child_bbox in enumerate(self.split_bbox(node.bbox)):
            if self.point_in_bbox(child_bbox, point):
                return self.sdf_recursive(point, node.children[idx])

        # If the point is outside all children, return the node's sdf_value
        return node.sdf_value

    def iterate_bbox_points(self, bbox):
        min_x, max_x, min_y, max_y, min_z, max_z = bbox
        for x in np.linspace(min_x, max_x, num=self.bbox_samples):
            for y in np.linspace(min_y, max_y, num=self.bbox_samples):
                for z in np.linspace(min_z, max_z, num=self.bbox_samples):
                    yield np.array([x, y, z])

    @staticmethod
    def point_in_bbox(bbox, point):
        min_x, max_x, min_y, max_y, min_z, max_z = bbox
        return (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y) and (min_z <= point[2] <= max_z)
