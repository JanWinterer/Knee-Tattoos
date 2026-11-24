"""Tracker tests: the contract on EVERY registered tracker, the registry, detect.py's checks, ArUco extras.

A new tracker file in trackers/ is picked up automatically. If it implements
render(), it is checked end to end: render a marker -> detect -> right id and centre.
"""
import json
import py_compile
from pathlib import Path

import numpy as np
import pytest

from marker_tracking import detect
from marker_tracking.io import Camera
from marker_tracking.trackers import (IMPORT_ERRORS, REGISTRY, Detection, Tracker, get_tracker,
                                      list_trackers, register)
from marker_tracking.trackers.aruco import CORNER_COLUMNS, PNP_COLUMNS
from marker_tracking.trackers.geometry import quad_centre

W, H = 200, 160
CAM = Camera(width=W, height=H, fps=30, fx=200.0, fy=200.0, cx=99.5, cy=79.5, depth_scale_m=0.001,
             source='synthetic')
MARKER_PX = 72
NO_DEPTH = np.zeros((H, W), np.float32)
TEMPLATE = Path(__file__).parents[1] / 'marker_tracking' / 'trackers' / '_template.py'


def marker_image(tracker, marker_id, x0, y0, size_px=MARKER_PX, width=W, height=H):
    """White BGR image with one rendered marker, top-left pixel at (x0, y0).

    Returns (image, true centre (u, v)). Pixel centres are integers, so a marker
    covering pixels x0 .. x0 + size_px - 1 has its centre at x0 + (size_px - 1) / 2.
    """
    grey = np.full((height, width), 255, np.uint8)
    grey[y0:y0 + size_px, x0:x0 + size_px] = tracker.render(marker_id, size_px)
    centre = (x0 + (size_px - 1) / 2, y0 + (size_px - 1) / 2)
    return np.dstack([grey] * 3), centre


# ---- contract, run on every registered tracker ----

@pytest.mark.parametrize('name', list_trackers())
def test_tracker_finds_its_own_marker(name):
    tracker = get_tracker(name)
    marker_id = 3 if tracker.max_id() is None else min(3, tracker.max_id())
    try:
        image, (u, v) = marker_image(tracker, marker_id, 50, 40)
    except NotImplementedError:
        pytest.skip(f'{name} cannot render markers')

    tracker.start(CAM, None)
    kept, dropped = detect.check_detections(tracker.detect(image, NO_DEPTH), tracker, W, H)  # raises on bad output
    assert [d.marker_id for d in kept] == [marker_id] and dropped == []
    assert (kept[0].u_px, kept[0].v_px) == (pytest.approx(u, abs=0.5), pytest.approx(v, abs=0.5))


@pytest.mark.parametrize('name', list_trackers())
def test_tracker_finds_nothing_in_a_blank_image(name):
    tracker = get_tracker(name)
    tracker.start(CAM, None)
    assert tracker.detect(np.full((H, W, 3), 255, np.uint8), NO_DEPTH) == []


@pytest.mark.parametrize('name', list_trackers())
def test_settings_can_be_stored_in_run_json(name):
    json.dumps(get_tracker(name).settings())


# ---- registry ----

def test_all_tracker_modules_were_imported():
    assert IMPORT_ERRORS == {}


def test_aruco_trackers_are_registered():
    assert {'aruco_4x4_50', 'aruco_5x5_50', 'aruco_6x6_50', 'aruco_7x7_50', 'aruco_original'} <= set(list_trackers())


def test_unknown_tracker():
    with pytest.raises(ValueError, match='unknown tracker'):
        get_tracker('no_such_tracker')


def test_get_tracker_sets_name_and_passes_settings():
    tracker = get_tracker('aruco_4x4_50', refinement='none')
    assert tracker.name == 'aruco_4x4_50'
    assert tracker.settings() == {'dictionary': 'DICT_4X4_50', 'refinement': 'none'}


def test_bad_setting_is_rejected():
    with pytest.raises(ValueError, match='refinement'):
        get_tracker('aruco_4x4_50', refinement='magic')


@pytest.fixture
def scratch_registry():
    """Names registered during a test; removed again afterwards so other tests see the normal registry."""
    names = []
    yield names
    for name in names:
        REGISTRY.pop(name, None)


def test_register_as_class_decorator(scratch_registry):
    @register('test_dummy')
    class Dummy(Tracker):
        def detect(self, color, depth_m):
            return []
    scratch_registry.append('test_dummy')

    assert isinstance(get_tracker('test_dummy'), Dummy)  # decorator hands the class back unchanged
    assert 'test_dummy' in list_trackers()


def test_name_registered_twice_fails():
    with pytest.raises(ValueError, match='registered twice'):
        register('aruco_4x4_50', Tracker)


def test_template_is_valid_python():
    py_compile.compile(str(TEMPLATE), doraise=True)  # compiled, not imported: importing would register it


# ---- detect.py: contract checks ----

class Scripted(Tracker):
    """Returns the detections it was given; for testing the contract checks."""
    name = 'scripted'
    extra_columns = ['score']

    def __init__(self, detections):
        self.detections = detections

    def detect(self, color, depth_m):
        return self.detections


def check(detections):
    return detect.check_detections(detections, Scripted(detections), W, H)


@pytest.mark.parametrize('detections, message', [
    ([Detection(-1, 10.0, 10.0)], 'integer >= 0'),
    ([Detection(1.5, 10.0, 10.0)], 'integer >= 0'),
    ([Detection(True, 10.0, 10.0)], 'integer >= 0'),
    ([Detection(1, float('nan'), 10.0)], 'non-finite'),
    ([Detection(1, W + 5.0, 10.0)], 'outside'),
    ([Detection(1, 10.0, -3.0)], 'outside'),
    ([Detection(1, 10.0, 10.0, {'speed': 1.0})], 'not listed in extra_columns'),
    ([Detection(1, 10.0, 10.0, {'score': 'high'})], 'not a number'),
    ([(1, 10.0, 10.0)], 'Detection objects'),
])
def test_contract_violations_raise(detections, message):
    with pytest.raises(detect.TrackerError, match=message):
        check(detections)


def test_detect_must_return_a_list():
    with pytest.raises(detect.TrackerError, match='must return a list'):
        detect.check_detections(None, Scripted(None), W, H)


def test_same_id_twice_is_dropped():
    kept, dropped = check([Detection(1, 10.0, 10.0), Detection(1, 50.0, 50.0), Detection(2, 30.0, 30.0)])
    assert [d.marker_id for d in kept] == [2]
    assert dropped == [1]


def test_numpy_values_become_plain_python():
    (d,), _ = check([Detection(np.int32(4), np.float32(1.5), np.float64(2.5), {'score': np.float32(0.25)})])
    assert type(d.marker_id) is int and type(d.u_px) is float and type(d.extra['score']) is float
    assert (d.marker_id, d.u_px, d.v_px, d.extra) == (4, 1.5, 2.5, {'score': 0.25})


def test_image_edges_count_as_inside():
    kept, _ = check([Detection(1, -0.5, -0.5), Detection(2, W - 0.5, H - 0.5)])
    assert len(kept) == 2


# ---- geometry ----

def test_quad_centre_is_the_true_centre_under_perspective():
    # A square in a plane tilted away from the camera: the far half looks smaller.
    corners_3d = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float) * 0.02
    tilt = np.radians(50)
    R = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    t = np.array([0.01, 0.0, 0.3])

    def project(p):
        x, y, z = R @ p + t
        return CAM.fx * x / z + CAM.cx, CAM.fy * y / z + CAM.cy

    corners = np.array([project(p) for p in corners_3d])
    true_centre = project(np.zeros(3))
    assert quad_centre(corners) == pytest.approx(true_centre, abs=1e-9)
    assert np.abs(corners.mean(axis=0) - true_centre).max() > 0.1  # the corner mean would be wrong


# ---- ArUco extras ----

def test_aruco_extras_corners_and_pnp():
    tracker = get_tracker('aruco_4x4_50')
    image, _ = marker_image(tracker, 7, 50, 40)
    size_m = 0.03
    true_z = size_m * CAM.fx / MARKER_PX  # distance at which a 3 cm marker appears MARKER_PX wide
    tracker.start(CAM, {'tracker': 'aruco_4x4_50', 'ids': [7], 'size_m': size_m})
    (d,) = tracker.detect(image, NO_DEPTH)

    assert set(d.extra) == set(CORNER_COLUMNS + PNP_COLUMNS)
    # Corners clockwise from the marker's top-left; the outer edge of pixel x0 is at x0 - 0.5
    assert (d.extra['c0_u_px'], d.extra['c0_v_px']) == (pytest.approx(49.5, abs=0.3), pytest.approx(39.5, abs=0.3))
    assert (d.extra['c2_u_px'], d.extra['c2_v_px']) == (pytest.approx(121.5, abs=0.3), pytest.approx(111.5, abs=0.3))
    assert d.extra['pnp_z_cam_m'] == pytest.approx(true_z, rel=0.01)


def test_aruco_without_marker_size_has_no_pnp():
    tracker = get_tracker('aruco_4x4_50')
    image, _ = marker_image(tracker, 7, 50, 40)
    tracker.start(CAM, None)
    (d,) = tracker.detect(image, NO_DEPTH)
    assert set(d.extra) == set(CORNER_COLUMNS)


def test_aruco_ignores_markers_of_another_dictionary():
    image, _ = marker_image(get_tracker('aruco_6x6_50'), 7, 50, 40)
    tracker = get_tracker('aruco_4x4_50')
    tracker.start(CAM, None)
    assert tracker.detect(image, NO_DEPTH) == []


def test_aruco_render_rejects_ids_out_of_range():
    with pytest.raises(ValueError, match='not in 0..49'):
        get_tracker('aruco_4x4_50').render(50, MARKER_PX)
