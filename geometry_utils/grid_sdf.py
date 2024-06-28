import trimesh
import time
import numpy as np
from geometry_utils.collision_detection import ObjectCollisionDetection
from solvers import Scene
from geometry_utils import bounding_boxes


EDGE_SAMPLES = 10
GRID_RESOLUTION = 10


class ObjectCollisionDetectionGrid(ObjectCollisionDetection):
    def __init__(self, obstacles, robot, offset=0.05, edge_samples=10, grid_resolution=10):
        super().__init__(obstacles, robot, offset)
        self.sdf_grids = [None]*len(obstacles)
        self.edge_samples = edge_samples
        self.grid_resolution = grid_resolution
        self.set_grid_sdf_grids()

    def sdf(self, obstacle_idx, point):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_boxes[obstacle_idx]
        grid_resolution = self.grid_resolution
        x, y, z = point[0], point[1], point[2]

        if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
            return np.inf

        # Calculate the size of each grid cell
        cell_size = (xmax - xmin) / (grid_resolution - 1), (ymax - ymin) / (grid_resolution - 1), (zmax - zmin) / (grid_resolution - 1)

        # Find the indices of the grid points surrounding the point
        i0 = 0 if cell_size[0] == 0 else int((x - xmin) / cell_size[0])
        j0 = 0 if cell_size[1] == 0 else int((y - ymin) / cell_size[1])
        k0 = 0 if cell_size[2] == 0 else int((z - zmin) / cell_size[2])

        i1 = min(i0 + 1, grid_resolution - 1)
        j1 = min(j0 + 1, grid_resolution - 1)
        k1 = min(k0 + 1, grid_resolution - 1)

        # Compute the relative positions within the cell
        xd = 0 if cell_size[0] == 0 else(x - (xmin + i0 * cell_size[0])) / cell_size[0]
        yd = 0 if cell_size[1] == 0 else (y - (ymin + j0 * cell_size[1])) / cell_size[1]
        zd = 0 if cell_size[2] == 0 else (z - (zmin + k0 * cell_size[2])) / cell_size[2]

        # Trilinear interpolation
        sdf_grid = self.sdf_grids[obstacle_idx]
        c00 = (1 - xd) * sdf_grid[i0, j0, k0] + xd * sdf_grid[i1, j0, k0]
        c01 = (1 - xd) * sdf_grid[i0, j0, k1] + xd * sdf_grid[i1, j0, k1]
        c10 = (1 - xd) * sdf_grid[i0, j1, k0] + xd * sdf_grid[i1, j1, k0]
        c11 = (1 - xd) * sdf_grid[i0, j1, k1] + xd * sdf_grid[i1, j1, k1]

        c0 = (1 - yd) * c00 + yd * c10
        c1 = (1 - yd) * c01 + yd * c11

        sdf_value = c0 * (1 - zd) + c1 * zd

        return sdf_value

    def set_grid_sdf_grids(self):
        grid_resolution = self.grid_resolution
        start_time = time.time()
        for obstacle_idx in range(len(self.obstacles)):
            if time.time() - self.start_time > 1200:
                return False
            obstacle = self.obstacles[obstacle_idx]
            bounding_box = self.bounding_boxes[obstacle_idx]
            # Create the grid
            x = np.linspace(bounding_box[0], bounding_box[1], grid_resolution)
            y = np.linspace(bounding_box[2], bounding_box[3], grid_resolution)
            z = np.linspace(bounding_box[4], bounding_box[5], grid_resolution)
            grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
            self.sdf_grids[obstacle_idx] = grid_points
            # Compute SDF for all grid points
            sdf_values = np.array([self.sdf_for_sample_point(obstacle.poly, p) for p in grid_points])
            self.sdf_grids[obstacle_idx] = sdf_values.reshape((grid_resolution, grid_resolution, grid_resolution))