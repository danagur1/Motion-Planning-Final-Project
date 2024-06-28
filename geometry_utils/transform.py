import numpy as np

def offset_polygon(polygon, offset):
    """
    Offset a CGAL Polygon_2 with a given CGAL Point_2 x,y offset

    :param polygon: polygon to offset
    :type polygon: :class:`numpy.array`
    :param offset: offset amount
    :type offset: :class:`numpy.array`

    :return: new offseted polygon
    :rtype: :class:`numpy.array`
    """
    new_points = []
    for point in polygon.vertices:
        new_points.append(np.array([point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]]))
    return np.array(new_points)