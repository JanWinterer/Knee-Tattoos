from dataclasses import replace

import cv2
import numpy as np
import pytest

from marker_tracking.io import Camera
from marker_tracking.lift import MIN_DEPTH_PX, lift, sample_depth

W, H = 64, 48
CAM = Camera(width=W, height=H, fps=30, fx=60.0, fy=60.0, cx=31.5, cy=23.5, depth_scale_m=0.001)


def wall(z=0.8):
    """Flat wall facing the camera at distance z [m]."""
    return np.full((H, W), z, np.float32)


def test_flat_wall_gives_exact_xyz():
    r = lift(wall(0.8), 40.0, 10.0, CAM)
    assert r['valid'] and r['n_depth_px'] == 49  # full 7 x 7 window
    expected = ((40.0 - 31.5) / 60 * 0.8, (10.0 - 23.5) / 60 * 0.8, 0.8)
    assert (r['x_cam_m'], r['y_cam_m'], r['z_cam_m']) == pytest.approx(expected, abs=1e-6)
    assert (r['u_px'], r['v_px']) == (40.0, 10.0)


def test_median_ignores_single_bad_pixels():
    depth = wall(0.8)
    depth[10, 40] = 5.0
    depth[11, 41] = 0.01
    assert lift(depth, 40.0, 10.0, CAM)['z_cam_m'] == pytest.approx(0.8)


def test_no_depth_means_not_valid():
    depth = wall(0.8)
    depth[5:16, 35:46] = 0  # hole around the marker
    r = lift(depth, 40.0, 10.0, CAM)
    assert not r['valid'] and r['n_depth_px'] == 0
    assert (r['x_cam_m'], r['y_cam_m'], r['z_cam_m']) == (0.0, 0.0, 0.0)
    assert (r['u_px'], r['v_px']) == (40.0, 10.0)  # the 2D detection is kept


def test_too_few_depth_pixels_means_not_valid():
    depth = np.zeros((H, W), np.float32)
    depth[10, 37:37 + MIN_DEPTH_PX - 1] = 0.8  # one pixel short
    r = lift(depth, 40.0, 10.0, CAM)
    assert r['n_depth_px'] == MIN_DEPTH_PX - 1 and not r['valid']


def test_nan_depth_is_ignored():
    depth = wall(0.8)
    depth[10, 40] = np.nan
    assert sample_depth(depth, 40.0, 10.0) == (pytest.approx(0.8), 48)


def test_window_is_clipped_at_the_image_border():
    assert sample_depth(wall(), 0.0, 0.0)[1] == 16           # 4 x 4 pixels left of a 7 x 7 window
    assert sample_depth(wall(), W - 1.0, H - 1.0)[1] == 16


def test_lens_distortion_is_undone():
    cam = replace(CAM, dist_coeffs=[0.1, -0.05, 0.001, 0.002, 0.0])
    point = np.array([0.1, -0.05, 0.7])
    pixel, _ = cv2.projectPoints(point.reshape(1, 1, 3), np.zeros(3), np.zeros(3), cam.K,
                                 np.array(cam.dist_coeffs))
    u, v = pixel.reshape(2)
    r = lift(wall(0.7), u, v, cam)
    assert (r['x_cam_m'], r['y_cam_m'], r['z_cam_m']) == pytest.approx(tuple(point), abs=1e-6)
