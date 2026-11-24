"""End to end: a tiny take with real ArUco markers -> run.py -> CSVs, extras and run.json."""
import json

import numpy as np
import pytest

from marker_tracking import run
from marker_tracking.check_take import check_take
from marker_tracking.io import Camera, write_take_camera, write_take_frame, write_take_markers
from marker_tracking.table import read_csv
from marker_tracking.trackers import get_tracker

CAM = Camera(width=320, height=240, fps=30, fx=300.0, fy=300.0, cx=159.5, cy=119.5, depth_scale_m=0.0001,
             source='synthetic')
TRACKER = 'aruco_4x4_50'
MARKER_PX = 60
WALL_M = 0.6                                  # flat wall facing the camera
SIZE_M = MARKER_PX * WALL_M / CAM.fx          # real marker size that appears MARKER_PX wide at WALL_M
TOP_LEFT = {0: (40, 50), 1: (200, 120)}       # marker id -> top-left pixel
N_FRAMES = 3


def make_marker_take(folder, markers_json_ids=(0, 1, 2)):
    """3 frames of a wall at 0.6 m with ArUco markers 0 and 1; marker 1 is hidden in frame 2.

    markers.json lists markers_json_ids (default 0, 1, 2: marker 2 is on the body but never seen);
    None = no markers.json.
    """
    tracker = get_tracker(TRACKER)
    write_take_camera(folder, CAM)
    if markers_json_ids is not None:
        write_take_markers(folder, TRACKER, markers_json_ids, SIZE_M)
    for i in range(N_FRAMES):
        grey = np.full((CAM.height, CAM.width), 255, np.uint8)
        for marker_id, (x0, y0) in TOP_LEFT.items():
            if marker_id == 1 and i == 2:
                continue  # occluded
            grey[y0:y0 + MARKER_PX, x0:x0 + MARKER_PX] = tracker.render(marker_id, MARKER_PX)
        write_take_frame(folder, i, np.dstack([grey] * 3), np.full(grey.shape, WALL_M), CAM.depth_scale_m)
    return folder


def true_xyz(marker_id):
    """Marker centre in the camera frame: pixel centre back-projected onto the wall."""
    x0, y0 = TOP_LEFT[marker_id]
    u, v = x0 + (MARKER_PX - 1) / 2, y0 + (MARKER_PX - 1) / 2
    return (u - CAM.cx) / CAM.fx * WALL_M, (v - CAM.cy) / CAM.fy * WALL_M, WALL_M


def rows_by_key(rows):
    return {(r['frame'], r['marker_id']): r for r in rows}


def test_test_take_is_a_valid_take(tmp_path):
    report = check_take(make_marker_take(tmp_path))
    assert report.ok, report.text()
    assert f'tracker {TRACKER}, ids [0, 1, 2]' in report.text()


def test_full_run(tmp_path, capsys):
    take, out = make_marker_take(tmp_path / 'take'), tmp_path / 'out'
    run.main(['--take', str(take), '--out', str(out)])  # tracker from markers.json
    assert 'detected 5 of 9' in capsys.readouterr().out

    camera = rows_by_key(read_csv(out / run.CAMERA_CSV))
    assert len(camera) == N_FRAMES * 3  # every listed marker in every frame
    for frame in range(N_FRAMES):
        r = camera[(frame, 0)]
        assert r['detected'] == r['valid'] == 1
        assert (r['x_cam_m'], r['y_cam_m'], r['z_cam_m']) == pytest.approx(true_xyz(0), abs=0.001)
        assert camera[(frame, 1)]['detected'] == (1 if frame < 2 else 0)
        assert camera[(frame, 2)]['detected'] == 0

    world = rows_by_key(read_csv(out / run.WORLD_CSV))
    assert world[(0, 0)]['x_m'] == camera[(0, 0)]['x_cam_m']  # no world.json: identity
    assert world[(2, 1)]['occluded'] == 1

    extras = rows_by_key(read_csv(out / run.EXTRA_CSV))
    assert sorted(extras) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]  # one row per detection
    assert extras[(0, 0)]['pnp_z_cam_m'] == pytest.approx(WALL_M, rel=0.01)  # independent of depth

    info = json.loads((out / run.RUN_JSON).read_text())
    assert info['tracker'] == TRACKER and info['settings']['refinement'] == 'subpix'
    assert info['marker_ids'] == [0, 1, 2] and info['ids_from'] == 'markers.json'
    assert info['marker_size_m'] == pytest.approx(SIZE_M)


def test_without_markers_json_the_ids_come_from_the_detections(tmp_path):
    take = make_marker_take(tmp_path / 'take', markers_json_ids=None)
    _, _, markers, result = run.run(take, tmp_path / 'out', TRACKER)
    assert markers is None
    assert result.marker_ids == [0, 1] and result.ids_from == 'detections'
    extras = read_csv(tmp_path / 'out' / run.EXTRA_CSV)
    assert np.isnan(extras[0]['pnp_z_cam_m'])  # no marker size, so no PnP: empty cells


def test_ids_missing_from_markers_json_are_ignored(tmp_path):
    take = make_marker_take(tmp_path / 'take', markers_json_ids=[0])
    _, _, _, result = run.run(take, tmp_path / 'out')
    assert result.marker_ids == [0] and result.ignored_ids == [1]


def test_markers_json_of_another_tracker_is_not_used(tmp_path, capsys):
    take = make_marker_take(tmp_path / 'take')
    run.main(['--take', str(take), '--out', str(tmp_path / 'out'), '--tracker', 'aruco_5x5_50'])
    assert 'markers.json is for aruco_4x4_50' in capsys.readouterr().out
    info = json.loads((tmp_path / 'out' / run.RUN_JSON).read_text())
    assert info['ids_from'] == 'detections' and info['marker_ids'] == []  # 5x5 finds no 4x4 markers


def test_no_tracker_anywhere_is_an_error(tmp_path):
    take = make_marker_take(tmp_path / 'take', markers_json_ids=None)
    with pytest.raises(ValueError, match='no markers.json'):
        run.run(take, tmp_path / 'out')


def test_settings_from_the_command_line(tmp_path):
    take = make_marker_take(tmp_path / 'take')
    run.main(['--take', str(take), '--out', str(tmp_path / 'out'), '--set', 'refinement=none'])
    info = json.loads((tmp_path / 'out' / run.RUN_JSON).read_text())
    assert info['settings']['refinement'] == 'none'


def test_parse_settings():
    assert run.parse_settings(['a=1', 'b=0.5', 'c=true', 'd=text']) == {'a': 1, 'b': 0.5, 'c': True, 'd': 'text'}
    with pytest.raises(ValueError, match='KEY=VALUE'):
        run.parse_settings(['oops'])


def test_list(capsys):
    run.main(['--list'])
    assert TRACKER in capsys.readouterr().out.split()
