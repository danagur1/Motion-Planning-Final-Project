import numpy as np


def Polyhedron_3_to_array_of_points(poly):
    """
    Convert a trimesh Polyhedron to list of points

    :param poly: polygon to convert
    :type poly: :class:`~trimesh.base.Trimesh`

    :return: list of tuples of (x,y, z) points
    :rtype: list<(:class:`float`, :class:`float`, :class:'float')>
    """
    return [(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])) for p in poly.vertices()]


def Point_3_to_xyz(point):
    """
    Convert trimesh.caching.TrackedArray to (x,y,z) values

    :param point: point to convert
    :type point: :class:`~trimesh.caching.TrackedArray`

    :return: x, y, z values of point
    :rtype: (:class:`float`, :class:`float`, class:'float')
    """
    return float(point[0]), float(point[1]), float(point[2]), float(point[3]), float(point[4]), float(point[5])


def Point_3_list_to_Point_d(point_3_list):
    """
    Convert a list of Point_3's to a high-dimensional Point_d

    :param point_3_list: list of Point_3's
    :type point_3_list: list<:class:`~trimesh.caching.TrackedArray`>

    :return: high dimensional point
    :rtype: :class:`~np.array`
    """
    coords = []
    for point in point_3_list:
        if type(point) == tuple:
            point = point[0]
        coords.append(point[0])
        coords.append(point[1])
        coords.append(point[2])
        coords.append(point[3])
        coords.append(point[4])
        coords.append(point[5])
    return np.array(coords)


def Point_d_to_Point_3_list(point_d):
    d = len(point_d)
    assert (d % 6 == 0)
    res = []
    for i in range(d // 6):
        res.append(np.array([point_d[6 * i], point_d[6 * i + 1], point_d[6 * i + 2], point_d[6 * i + 3],
                             point_d[6 * i + 4], point_d[6 * i + 5]]))
    return res
