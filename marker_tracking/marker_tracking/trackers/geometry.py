"""Geometry helpers shared by trackers (this module registers no tracker)."""
import numpy as np


def quad_centre(corners):
    """Centre of a square marker seen in perspective: the intersection of its two diagonals.

    The mean of the 4 corners is not the centre once the marker is tilted
    (perspective shrinks the far half). The diagonals of a square meet at its
    true centre in every view, because perspective keeps straight lines straight.

    corners: (4, 2) pixel coordinates, in order around the square.
    Returns (u, v) as a (2,) array.
    """
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    p0, p1, p2, p3 = (np.append(p, 1.0) for p in c)  # homogeneous coordinates
    diagonal_02 = np.cross(p0, p2)                    # line through corners 0 and 2
    diagonal_13 = np.cross(p1, p3)                    # line through corners 1 and 3
    x, y, w = np.cross(diagonal_02, diagonal_13)      # their intersection
    if abs(w) < 1e-12:
        return c.mean(axis=0)  # diagonals parallel: degenerate quad, best effort
    return np.array([x / w, y / w])
