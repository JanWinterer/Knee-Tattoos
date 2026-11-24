import json

import cv2
import numpy as np
import pytest

from marker_tracking import frame
from marker_tracking.io import (Camera, depth_to_png_values, load_take, read_markers, write_take_camera,
                                write_take_frame, write_take_markers)

W, H = 64, 48
CAM = Camera(width=W, height=H, fps=30, fx=60.0, fy=60.0, cx=31.5, cy=23.5, depth_scale_m=0.0001,
             source='synthetic')


def make_take(folder, n=3, cam=CAM, T=None):
    """Tiny take: frame i has colour value 10*i and a flat wall at 0.5 + 0.1*i m, with a hole at (0, 0)."""
    write_take_camera(folder, cam, T, 'plate' if T is not None else 'identity')
    for i in range(n):
        color = np.full((cam.height, cam.width, 3), 10 * i, np.uint8)
        depth = np.full((cam.height, cam.width), 0.5 + 0.1 * i)
        depth[0, 0] = np.nan
        write_take_frame(folder, i, color, depth, cam.depth_scale_m)
    return folder


def test_round_trip(tmp_path):
    take = load_take(make_take(tmp_path))
    assert take.n_frames == len(take) == 3
    assert take.camera == CAM
    assert np.array_equal(take.camera.K, [[60, 0, 31.5], [0, 60, 23.5], [0, 0, 1]])

    f = take.frame(2)
    assert (f.index, f.time_s) == (2, pytest.approx(2 / 30))
    assert f.color.shape == (H, W, 3) and f.color.dtype == np.uint8 and f.color[5, 5, 0] == 20
    assert f.depth_m.shape == (H, W) and f.depth_m.dtype == np.float32
    assert f.depth_m[5, 5] == pytest.approx(0.7, abs=1e-4)
    assert f.depth_m[0, 0] == 0  # NaN written as "no depth"

    assert [f.index for f in take] == [0, 1, 2]


def test_no_world_json_is_identity(tmp_path):
    take = load_take(make_take(tmp_path))
    assert np.array_equal(take.T_world_cam, np.eye(4))
    assert take.world_source == 'identity'


def test_world_json_is_read(tmp_path):
    T = frame.make_transform(t=[0, 0, 1])
    take = load_take(make_take(tmp_path, T=T))
    assert np.allclose(take.T_world_cam, T)
    assert take.world_source == 'plate'


def test_camera_json_is_readable_by_hand(tmp_path):
    make_take(tmp_path)
    data = json.loads((tmp_path / 'camera.json').read_text())
    assert data['format_version'] == 1
    assert data['fx'] == 60.0 and data['depth_scale_m'] == 0.0001


def test_depth_png_is_16_bit(tmp_path):
    make_take(tmp_path)
    raw = cv2.imread(str(tmp_path / 'depth' / '000001.png'), cv2.IMREAD_UNCHANGED)
    assert raw.dtype == np.uint16 and raw[5, 5] == 6000  # 0.6 m / 0.1 mm


def test_depth_conversion_edge_cases():
    assert depth_to_png_values([0.0, -1.0, np.nan, np.inf, 0.0012], 0.001).tolist() == [0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        depth_to_png_values([7.0], 0.0001)  # 70000 units > 16 bit


def test_missing_camera_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_take(tmp_path)


def test_wrong_format_version(tmp_path):
    make_take(tmp_path)
    data = json.loads((tmp_path / 'camera.json').read_text())
    data['format_version'] = 2
    (tmp_path / 'camera.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='format_version'):
        load_take(tmp_path)


def test_missing_depth_frame(tmp_path):
    make_take(tmp_path)
    (tmp_path / 'depth' / '000001.png').unlink()
    with pytest.raises(ValueError, match='different frame numbers'):
        load_take(tmp_path)


def test_gap_in_frames(tmp_path):
    make_take(tmp_path)
    for d in ('color', 'depth'):
        (tmp_path / d / '000001.png').unlink()
    with pytest.raises(ValueError, match='without gaps'):
        load_take(tmp_path)


def test_frame_out_of_range(tmp_path):
    with pytest.raises(IndexError):
        load_take(make_take(tmp_path)).frame(3)


def test_writer_rejects_bad_input(tmp_path):
    write_take_camera(tmp_path, CAM)
    with pytest.raises(ValueError):
        write_take_frame(tmp_path, 0, np.zeros((H, W), np.uint8), np.ones((H, W)), 0.001)  # grey
    with pytest.raises(ValueError):
        write_take_frame(tmp_path, 0, np.zeros((H, W, 3), np.uint8), np.ones((H, W + 1)), 0.001)  # size


# ---- markers.json ----

def test_markers_json_round_trip(tmp_path):
    make_take(tmp_path)
    write_take_markers(tmp_path, 'aruco_4x4_50', [3, 0, 1], size_m=0.03)
    assert load_take(tmp_path).markers == {'tracker': 'aruco_4x4_50', 'ids': [0, 1, 3], 'size_m': 0.03}


def test_markers_json_is_optional(tmp_path):
    assert load_take(make_take(tmp_path)).markers is None


def test_markers_json_without_size(tmp_path):
    make_take(tmp_path)
    write_take_markers(tmp_path, 'aruco_4x4_50', [0])
    assert load_take(tmp_path).markers['size_m'] is None


@pytest.mark.parametrize('content, message', [
    ({'ids': [0]}, "'tracker'"),
    ({'tracker': 'aruco_4x4_50'}, "'ids'"),
    ({'tracker': 'aruco_4x4_50', 'ids': []}, "'ids'"),
    ({'tracker': 'aruco_4x4_50', 'ids': [0, -1]}, "'ids'"),
    ({'tracker': 'aruco_4x4_50', 'ids': [0, 1.5]}, "'ids'"),
    ({'tracker': 'aruco_4x4_50', 'ids': [0, 0]}, 'duplicates'),
    ({'tracker': 'aruco_4x4_50', 'ids': [0], 'size_m': 0}, "'size_m'"),
])
def test_bad_markers_json(tmp_path, content, message):
    make_take(tmp_path)
    (tmp_path / 'markers.json').write_text(json.dumps(content))
    with pytest.raises(ValueError, match=message):
        read_markers(tmp_path / 'markers.json')
