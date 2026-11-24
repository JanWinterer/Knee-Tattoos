"""ArUco markers: square black-and-white markers carrying a binary id code (OpenCV aruco module).

Registered once per dictionary. The name gives the code grid and the number of ids:
    aruco_4x4_50     4 x 4 data cells, ids 0..49
    aruco_5x5_50     5 x 5 data cells, ids 0..49
    aruco_6x6_50     6 x 6 data cells, ids 0..49
    aruco_7x7_50     7 x 7 data cells, ids 0..49
    aruco_original   5 x 5 data cells, ids 0..1023 (the original ArUco dictionary)
Fewer cells = larger cells at the same marker size (easier to read when small),
but weaker error correction. Every marker also has a one-cell black border.

Setting (get_tracker(name, refinement=...)):
    refinement   corner refinement: 'subpix' (default), 'contour' or 'none'

Extras (markers_extra.csv):
    c0_u_px ... c3_v_px   the 4 corners, clockwise from the marker's own top-left corner
    pnp_x/y/z_cam_m       marker centre from the corners and the known marker size (PnP),
    pnp_rx/ry/rz_rad      and its rotation (Rodrigues vector), camera frame.
                          Only if markers.json gives size_m. Depth is not used, so this
                          is an independent second estimate next to the depth-based xyz.
"""
import functools

import cv2
import numpy as np

from marker_tracking.trackers.base import Detection, Tracker, register
from marker_tracking.trackers.geometry import quad_centre

CORNER_COLUMNS = [f'c{i}_{axis}_px' for i in range(4) for axis in ('u', 'v')]
PNP_COLUMNS = ['pnp_x_cam_m', 'pnp_y_cam_m', 'pnp_z_cam_m', 'pnp_rx_rad', 'pnp_ry_rad', 'pnp_rz_rad']

REFINEMENT_METHODS = {
    'none': cv2.aruco.CORNER_REFINE_NONE,
    'subpix': cv2.aruco.CORNER_REFINE_SUBPIX,
    'contour': cv2.aruco.CORNER_REFINE_CONTOUR,
}

# Registered name -> OpenCV dictionary constant (name in cv2.aruco)
DICTIONARIES = {
    'aruco_4x4_50': 'DICT_4X4_50',
    'aruco_5x5_50': 'DICT_5X5_50',
    'aruco_6x6_50': 'DICT_6X6_50',
    'aruco_7x7_50': 'DICT_7X7_50',
    'aruco_original': 'DICT_ARUCO_ORIGINAL',
}


class ArucoTracker(Tracker):
    extra_columns = CORNER_COLUMNS + PNP_COLUMNS

    def __init__(self, dictionary, refinement='subpix'):
        if refinement not in REFINEMENT_METHODS:
            raise ValueError(f'refinement {refinement!r}, choose from {sorted(REFINEMENT_METHODS)}')
        self.dictionary_name = dictionary
        self.refinement = refinement
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = REFINEMENT_METHODS[refinement]
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)
        self.size_m = None  # outer side length of the black square, from markers.json

    def start(self, camera, markers):
        super().start(camera, markers)
        self.size_m = markers.get('size_m') if markers else None

    def settings(self):
        return {'dictionary': self.dictionary_name, 'refinement': self.refinement}

    def max_id(self):
        return self.dictionary.bytesList.shape[0] - 1

    def detect(self, color, depth_m):
        grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(grey)
        if ids is None:
            return []

        detections = []
        for quad, marker_id in zip(corners, ids.ravel()):
            quad = quad.reshape(4, 2).astype(float)  # clockwise from the marker's top-left corner
            u, v = quad_centre(quad)
            extra = dict(zip(CORNER_COLUMNS, quad.ravel()))
            if self.size_m:
                extra.update(self._pose_from_corners(quad))
            detections.append(Detection(int(marker_id), float(u), float(v), extra))
        return detections

    def _pose_from_corners(self, quad):
        """Marker pose from its 4 corners and known size (PnP). Returns PNP_COLUMNS values, or {} on failure."""
        half = self.size_m / 2
        # Corners in the marker's own frame, in the order SOLVEPNP_IPPE_SQUARE expects
        # (same order as the detected corners: top-left, top-right, bottom-right, bottom-left).
        marker_corners = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        ok, rvec, tvec = cv2.solvePnP(marker_corners, quad, self.camera.K,
                                      np.asarray(self.camera.dist_coeffs, dtype=float),
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return {}
        return dict(zip(PNP_COLUMNS, np.concatenate([tvec.ravel(), rvec.ravel()])))

    def render(self, marker_id, size_px):
        """Black-on-white marker including its black border, without a white margin around it.

        Print it with a white margin of at least one cell, or detection fails.
        """
        if not 0 <= marker_id <= self.max_id():
            raise ValueError(f'{self.name}: marker id {marker_id} not in 0..{self.max_id()}')
        return cv2.aruco.generateImageMarker(self.dictionary, int(marker_id), int(size_px), borderBits=1)


for _name, _dictionary in DICTIONARIES.items():
    register(_name, functools.partial(ArucoTracker, _dictionary))
