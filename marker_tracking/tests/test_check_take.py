import json

import cv2
import numpy as np
import pytest

from marker_tracking.check_take import check_take, main
from marker_tracking.io import write_take_markers
from test_io import CAM, make_take


def errors_and_warnings(folder):
    r = check_take(folder)
    return ' | '.join(r.errors), ' | '.join(r.warnings)


def edit_camera(folder, **changes):
    data = json.loads((folder / 'camera.json').read_text())
    data.update(changes)
    (folder / 'camera.json').write_text(json.dumps(data))


def test_good_take_passes(tmp_path):
    r = check_take(make_take(tmp_path))
    assert r.ok and not r.warnings, r.text()
    assert 'frames: 3' in r.text()


def test_not_a_take(tmp_path):
    errors, _ = errors_and_warnings(tmp_path)
    assert 'camera.json' in errors


def test_bad_json(tmp_path):
    make_take(tmp_path)
    (tmp_path / 'camera.json').write_text('{ nope')
    assert 'not valid JSON' in errors_and_warnings(tmp_path)[0]


@pytest.mark.parametrize('changes, message', [
    ({'fx': 0}, 'focal lengths'),
    ({'fps': 0}, 'fps'),
    ({'depth_scale_m': 0}, 'depth_scale_m'),
    ({'cx': 500}, 'principal point'),
    ({'dist_coeffs': [0, 0, 0]}, 'dist_coeffs'),
    ({'format_version': 2}, 'format_version'),
    ({'width': 32}, 'camera.json says'),          # images bigger than camera.json says
])
def test_camera_errors(tmp_path, changes, message):
    make_take(tmp_path)
    edit_camera(tmp_path, **changes)
    assert message in errors_and_warnings(tmp_path)[0]


def test_missing_key(tmp_path):
    make_take(tmp_path)
    data = json.loads((tmp_path / 'camera.json').read_text())
    del data['cy']
    (tmp_path / 'camera.json').write_text(json.dumps(data))
    assert "missing keys ['cy']" in errors_and_warnings(tmp_path)[0]


@pytest.mark.parametrize('changes, message', [
    ({'fx': 2.0, 'fy': 2.0}, 'field of view'),    # focal length in mm instead of px
    ({'fy': 80.0}, 'differ a lot'),
    ({'cx': 5.0}, 'far from the image centre'),
])
def test_camera_warnings(tmp_path, changes, message):
    make_take(tmp_path)
    edit_camera(tmp_path, **changes)
    errors, warnings = errors_and_warnings(tmp_path)
    assert not errors and message in warnings


def test_wrong_depth_scale_is_noticed(tmp_path):
    # depth written in 0.1 mm but camera.json claims 1 um: median depth 0.005 m
    make_take(tmp_path)
    edit_camera(tmp_path, depth_scale_m=0.000001)
    assert 'wrong depth_scale_m' in errors_and_warnings(tmp_path)[1]


def test_missing_and_extra_frames(tmp_path):
    make_take(tmp_path)
    (tmp_path / 'depth' / '000002.png').unlink()
    (tmp_path / 'color' / '000001.png').unlink()
    errors, _ = errors_and_warnings(tmp_path)
    assert 'frames without depth: [2]' in errors
    assert 'frames without colour: [1]' in errors
    assert 'missing [1]' in errors


def test_depth_must_be_16_bit(tmp_path):
    make_take(tmp_path)
    cv2.imwrite(str(tmp_path / 'depth' / '000001.png'), np.full((CAM.height, CAM.width), 50, np.uint8))
    assert 'frame 1: depth must be 16-bit' in errors_and_warnings(tmp_path)[0]


def test_color_must_have_3_channels(tmp_path):
    make_take(tmp_path)
    cv2.imwrite(str(tmp_path / 'color' / '000000.png'), np.zeros((CAM.height, CAM.width), np.uint8))
    assert 'frame 0: colour must be 8-bit 3-channel' in errors_and_warnings(tmp_path)[0]


def test_sparse_depth_warning(tmp_path):
    make_take(tmp_path)
    cv2.imwrite(str(tmp_path / 'depth' / '000000.png'), np.zeros((CAM.height, CAM.width), np.uint16))
    errors, warnings = errors_and_warnings(tmp_path)
    assert not errors and '1 frames with depth on < 10%' in warnings


def test_bad_world_json(tmp_path):
    make_take(tmp_path)
    (tmp_path / 'world.json').write_text('{"T_world_cam": [[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}')
    assert 'world.json' in errors_and_warnings(tmp_path)[0]


def test_every_n_frames(tmp_path):
    make_take(tmp_path, n=5)
    assert 'checked 2' in check_take(tmp_path, every=3).text()


def test_main_exit_code(tmp_path, capsys):
    make_take(tmp_path / 'good')
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / 'good')])
    assert e.value.code == 0 and 'OK' in capsys.readouterr().out

    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / 'missing')])
    assert e.value.code == 1



# ---- markers.json ----

def test_markers_json_is_reported(tmp_path):
    make_take(tmp_path)
    write_take_markers(tmp_path, 'aruco_4x4_50', [0, 1], size_m=0.03)
    r = check_take(tmp_path)
    assert r.ok, r.text()
    assert 'markers.json: tracker aruco_4x4_50, ids [0, 1], 30 mm' in r.text()


def test_markers_json_unknown_tracker(tmp_path):
    make_take(tmp_path)
    write_take_markers(tmp_path, 'no_such_tracker', [0])
    assert 'unknown tracker' in errors_and_warnings(tmp_path)[0]


def test_markers_json_id_the_tracker_cannot_produce(tmp_path):
    make_take(tmp_path)
    write_take_markers(tmp_path, 'aruco_4x4_50', [0, 60])
    assert 'ids [60] do not exist for aruco_4x4_50 (0..49)' in errors_and_warnings(tmp_path)[0]


def test_markers_json_broken(tmp_path):
    make_take(tmp_path)
    (tmp_path / 'markers.json').write_text('{"tracker": "aruco_4x4_50", "ids": "all"}')
    assert 'markers.json' in errors_and_warnings(tmp_path)[0]
