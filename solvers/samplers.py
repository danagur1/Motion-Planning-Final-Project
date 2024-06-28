import math
import random
from geometry_utils.bounding_boxes import calc_scene_bounding_box


class Sampler(object):
    """
    Abstract class for sampling methods in the scene.

    :param scene: a scene to sample in
    :type scene: :class:`~solvers.Scene`
    """
    def __init__(self, scene=None):
        self.scene = scene
        if self.scene is not None:
            self.set_scene(self.scene)

    def set_scene(self, scene):
        """
        Set the scene the sampler should use.
        Can be overridded to add additional processing.

        :param scene: a scene to sample in
        :type scene: :class:`~solvers.Scene`
        """
        self.scene = scene
    
    def sample(self):
        """
        Return a sample in the space (might be invalid)

        :return: sampled point
        :rtype: :class:`~numpy.array`
        """
        return None


class Sampler_Uniform(Sampler):
    """
    Uniform sampler in the scene

    :param scene: a scene to sample in
    :type scene: :class:`~solvers.Scene`
    """
    def __init__(self, scene=None):
        super().__init__(scene)
        if scene is None:
            self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = \
                None, None, None, None, None, None
            # remember scene bounds

    def set_bounds_manually(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """
        Set the sampling bounds manually (instead of supplying a scene)
        Bounds are given in CGAL :class:`~numpy.float64`
        """
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = min_x, max_x, min_y, max_y, min_z, max_z

    def set_scene(self, scene, bounding_box=None):
        """
        Set the scene the sampler should use.
        Can be overridded to add additional processing.

        :param scene: a scene to sample in
        :type scene: :class:`~solvers.Scene`
        """
        super().set_scene(scene)
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = \
            bounding_box or calc_scene_bounding_box(self.scene)

    def sample(self):
        """
        Return a sample in the space (might be invalid)

        :return: sampled point
        :rtype: :class:`~numpy.array`
        """
        x = random.uniform(self.min_x, self.max_x)
        y = random.uniform(self.min_y, self.max_y)
        z = random.uniform(self.min_z, self.max_z)
        angle_x = random.uniform(0, 2*math.pi)
        angle_y = random.uniform(0, 2*math.pi)
        angle_z = random.uniform(0, 2*math.pi)
        return [x, y, z, angle_x, angle_y, angle_z]
