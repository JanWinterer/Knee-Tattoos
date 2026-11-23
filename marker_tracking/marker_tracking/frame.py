"""Camera frame -> global (world) frame.

The camera measures every marker relative to itself. The rest of the pipeline
(and SKEL) wants positions in one fixed global frame. One rigid 4x4 transform,
T_world_cam, links the two:

    p_world = R @ p_cam + t        T_world_cam = [[R, t], [0, 0, 0, 1]]

Where T_world_cam comes from:
    now:   identity, i.e. global frame = camera frame
    later: the calibration plate pose (only the source changes, nothing else)

Typical use:
    T, source = load('world.json')     # or load(None) -> identity
    xyz_world = apply(T, xyz_cam)      # (3,) or (N, 3) or (..., 3)

world.json:
    {"T_world_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
     "source": "identity"}
"""
import json

import numpy as np

TOL = 1e-6  # allowed deviation from a perfect rotation / bottom row


def identity():
    """4x4 identity: global frame = camera frame."""
    return np.eye(4)


def make_transform(R=None, t=None):
    """Build a 4x4 transform from a 3x3 rotation R and a translation t [m]."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return check_transform(T)


def check_transform(T, tol=TOL):
    """Return T as a float (4, 4) array, or raise ValueError if it is not rigid.

    Rigid = rotation plus translation only: no scaling, shearing or mirroring.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f'transform must be 4x4, got shape {T.shape}')
    if not np.all(np.isfinite(T)):
        raise ValueError('transform contains NaN or inf')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
        raise ValueError(f'bottom row must be [0, 0, 0, 1], got {T[3].tolist()}')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        raise ValueError('rotation part is not orthonormal (scaled or sheared)')
    if np.linalg.det(R) < 0:
        raise ValueError('rotation part is a reflection (det < 0)')
    return T


def invert(T):
    """Inverse of a rigid transform, e.g. T_cam_world -> T_world_cam."""
    T = check_transform(T)
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def compose(*Ts):
    """Chain transforms left to right: compose(T_a_b, T_b_c) = T_a_c."""
    out = np.eye(4)
    for T in Ts:
        out = out @ check_transform(T)
    return out


def apply(T, points):
    """Transform points [m]. Accepts shape (3,), (N, 3) or (..., 3); returns the same shape."""
    T = check_transform(T)
    p = np.asarray(points, dtype=float)
    if p.shape[-1] != 3:
        raise ValueError(f'points must have 3 coordinates in the last axis, got shape {p.shape}')
    return p @ T[:3, :3].T + T[:3, 3]


def load(path=None):
    """Read T_world_cam from a JSON file. Returns (T, source).

    path None -> (identity, 'identity'). A path that does not exist is an error,
    so a typo never silently falls back to the camera frame.
    The file may hold other keys too (e.g. camera.json); only
    'T_world_cam' and the optional 'source' are read.
    """
    if path is None:
        return identity(), 'identity'
    with open(path) as f:
        data = json.load(f)
    if 'T_world_cam' not in data:
        raise ValueError(f"{path}: no 'T_world_cam' key")
    return check_transform(data['T_world_cam']), str(data.get('source', 'unknown'))


def save(path, T, source):
    """Write T_world_cam and its source ('identity', 'blender', 'plate', ...) to JSON."""
    T = check_transform(T)
    matrix = ',\n    '.join(json.dumps(row) for row in T.tolist())  # one matrix row per line
    with open(path, 'w') as f:
        f.write(f'{{\n  "T_world_cam": [\n    {matrix}\n  ],\n  "source": {json.dumps(source)}\n}}\n')
