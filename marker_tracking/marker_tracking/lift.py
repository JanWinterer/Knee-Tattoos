"""Sample depth at a marker centre and back-project to camera-frame xyz.

The same for every tracker, so all trackers are compared through the same depth camera.

    result = lift(depth_m, u, v, camera)      # camera: io.Camera
    result = {'u_px', 'v_px', 'x_cam_m', 'y_cam_m', 'z_cam_m', 'n_depth_px', 'valid'}
             (the per-detection dict that table.make_camera_rows expects)

Depth: the median of the non-zero depth pixels in a (2 * radius_px + 1)^2 window
around the centre. The median ignores single bad pixels and is exact on a flat
surface through the centre. Fewer than min_depth_px depth pixels -> valid = False, xyz = 0.

Back-projection (OpenCV camera, depth = z along the optical axis):
    (x_n, y_n) = undistorted normalised image coordinates of (u, v)
    x = x_n * z,   y = y_n * z
"""
import cv2
import numpy as np

RADIUS_PX = 3      # 7 x 7 pixel window
MIN_DEPTH_PX = 5   # fewer depth pixels in the window -> no 3D position


def sample_depth(depth_m, u, v, radius_px=RADIUS_PX):
    """Median depth [m] around pixel (u, v) and the number of depth pixels used (0 -> depth 0)."""
    height, width = depth_m.shape
    centre_u, centre_v = int(round(u)), int(round(v))
    # Window clipped to the image
    u0, u1 = max(centre_u - radius_px, 0), min(centre_u + radius_px + 1, width)
    v0, v1 = max(centre_v - radius_px, 0), min(centre_v + radius_px + 1, height)
    if u0 >= u1 or v0 >= v1:
        return 0.0, 0
    window = depth_m[v0:v1, u0:u1]
    good = window[np.isfinite(window) & (window > 0)]
    if good.size == 0:
        return 0.0, 0
    return float(np.median(good)), int(good.size)


def normalised(u, v, camera):
    """Pixel (u, v) -> undistorted normalised coordinates (x_n, y_n) on the plane z = 1."""
    pixel = np.array([[[u, v]]], dtype=np.float64)
    dist_coeffs = np.asarray(camera.dist_coeffs, dtype=np.float64)
    x_n, y_n = cv2.undistortPoints(pixel, camera.K, dist_coeffs).reshape(2)
    return float(x_n), float(y_n)


def back_project(u, v, z, camera):
    """Pixel (u, v) at depth z [m] -> (x, y, z) in the camera frame [m]."""
    x_n, y_n = normalised(u, v, camera)
    return x_n * z, y_n * z, z


def lift(depth_m, u, v, camera, radius_px=RADIUS_PX, min_depth_px=MIN_DEPTH_PX):
    """One detected marker centre -> result dict for table.make_camera_rows."""
    z, n_depth_px = sample_depth(depth_m, u, v, radius_px)
    result = {'u_px': float(u), 'v_px': float(v), 'n_depth_px': n_depth_px, 'valid': False,
              'x_cam_m': 0.0, 'y_cam_m': 0.0, 'z_cam_m': 0.0}
    if n_depth_px >= min_depth_px and z > 0:
        result['x_cam_m'], result['y_cam_m'], result['z_cam_m'] = back_project(u, v, z, camera)
        result['valid'] = True
    return result
