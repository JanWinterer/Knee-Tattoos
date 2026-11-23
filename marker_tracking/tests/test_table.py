import csv
import json
import os

import pytest

from marker_tracking import frame
from marker_tracking.camera_to_world import convert
from marker_tracking.table import (CAMERA_COLUMNS, WORLD_COLUMNS, make_camera_rows, make_world_rows,
                                   read_csv, write_camera_csv, write_world_csv)

EXAMPLES = os.path.join(os.path.dirname(__file__), '..', 'examples')
CAMERA_EXAMPLE = os.path.join(EXAMPLES, 'markers_camera_example.csv')
WORLD_EXAMPLE = os.path.join(EXAMPLES, 'markers_world_example.csv')

FOUND = {'u_px': 612.4, 'v_px': 355.1, 'x_cam_m': 0.3, 'y_cam_m': 0.0, 'z_cam_m': 0.4,
         'n_depth_px': 25, 'valid': True}
NO_DEPTH = {'u_px': 700.0, 'v_px': 401.3, 'x_cam_m': 9.0, 'y_cam_m': 9.0, 'z_cam_m': 9.0,
            'n_depth_px': 2, 'valid': False}
SHIFT = frame.make_transform(t=[1.0, 2.0, 3.0])


def read(path):
    with open(path, newline='') as f:
        return list(csv.reader(f))


# ---- camera table ----

def test_one_row_per_frame_and_marker_sorted():
    rows = make_camera_rows(3, 30, [2, 0, 1], {})
    assert len(rows) == 9
    assert [(r['frame'], r['marker_id']) for r in rows] == [(f, m) for f in range(3) for m in (0, 1, 2)]


def test_time_is_frame_over_fps():
    rows = make_camera_rows(31, 30, [0], {})
    assert rows[30]['time_s'] == 1.0


def test_flags_and_zeros():
    rows = make_camera_rows(1, 30, [0, 1, 2], {(0, 0): FOUND, (0, 1): NO_DEPTH})
    found, no_depth, missing = rows

    assert (found['detected'], found['valid'], found['z_cam_m']) == (1, 1, 0.4)

    # detected without depth: pixel position kept, xyz and dist zeroed even if the input had values
    assert (no_depth['detected'], no_depth['valid']) == (1, 0)
    assert (no_depth['u_px'], no_depth['n_depth_px']) == (700.0, 2)
    assert (no_depth['x_cam_m'], no_depth['y_cam_m'], no_depth['z_cam_m'], no_depth['dist_m']) == (0, 0, 0, 0)

    assert (missing['detected'], missing['valid']) == (0, 0)
    assert all(missing[c] == 0 for c in CAMERA_COLUMNS[5:])


def test_dist_is_distance_from_camera():
    (row,) = make_camera_rows(1, 30, [0], {(0, 0): FOUND})
    assert row['dist_m'] == pytest.approx(0.5)  # 3-4-5 triangle


def test_written_camera_csv_header_and_format(tmp_path):
    path = tmp_path / 'markers_camera.csv'
    write_camera_csv(make_camera_rows(1, 30, [0, 1], {(0, 0): FOUND}), path)
    lines = read(path)
    assert lines[0] == CAMERA_COLUMNS
    assert lines[1] == ['0', '0.000000', '0', '1', '1', '612.400', '355.100',
                        '0.300000', '0.000000', '0.400000', '0.500000', '25']
    assert lines[2] == ['0', '0.000000', '1', '0', '0', '0.000', '0.000',
                        '0.000000', '0.000000', '0.000000', '0.000000', '0']


def test_rerun_gives_identical_file(tmp_path):
    rows = make_camera_rows(5, 30, [0, 1], {(2, 1): FOUND, (3, 0): NO_DEPTH})
    write_camera_csv(rows, tmp_path / 'a.csv')
    write_camera_csv(rows, tmp_path / 'b.csv')
    assert (tmp_path / 'a.csv').read_bytes() == (tmp_path / 'b.csv').read_bytes()


# ---- world table ----

def test_world_rows_transform_valid_and_zero_the_rest():
    camera_rows = make_camera_rows(1, 30, [0, 1, 2], {(0, 0): FOUND, (0, 1): NO_DEPTH})
    found, no_depth, missing = make_world_rows(camera_rows, SHIFT)

    assert (found['x_m'], found['y_m'], found['z_m']) == pytest.approx((1.3, 2.0, 3.4))
    assert (found['occluded'], found['valid']) == (0, 1)

    # not valid: xyz 0, not shifted by the translation
    assert (no_depth['x_m'], no_depth['y_m'], no_depth['z_m']) == (0.0, 0.0, 0.0)
    assert (no_depth['occluded'], no_depth['valid']) == (0, 0)

    assert (missing['x_m'], missing['y_m'], missing['z_m']) == (0.0, 0.0, 0.0)
    assert (missing['occluded'], missing['valid']) == (1, 0)


def test_world_rows_identity_keeps_camera_xyz():
    camera_rows = make_camera_rows(1, 30, [0], {(0, 0): FOUND})
    (row,) = make_world_rows(camera_rows, frame.identity())
    assert (row['x_m'], row['y_m'], row['z_m']) == (0.3, 0.0, 0.4)


def test_world_rows_same_order_and_time():
    camera_rows = make_camera_rows(4, 30, [1, 0], {(2, 1): FOUND})
    world_rows = make_world_rows(camera_rows, SHIFT)
    keys = ['frame', 'time_s', 'marker_id']
    assert [[r[k] for k in keys] for r in world_rows] == [[r[k] for k in keys] for r in camera_rows]


def test_world_rows_empty():
    assert make_world_rows([], SHIFT) == []


def test_written_world_csv_and_sidecar(tmp_path):
    camera_rows = make_camera_rows(1, 30, [0, 1], {(0, 0): FOUND})
    path = tmp_path / 'markers_world.csv'
    write_world_csv(make_world_rows(camera_rows, SHIFT), path, SHIFT, 'plate')

    lines = read(path)
    assert lines[0] == WORLD_COLUMNS
    assert lines[1] == ['0', '0.000000', '0', '1.300000', '2.000000', '3.400000', '0', '1']
    assert lines[2] == ['0', '0.000000', '1', '0.000000', '0.000000', '0.000000', '1', '0']

    T, source = frame.load(tmp_path / 'markers_world.json')
    assert T == pytest.approx(SHIFT)
    assert source == 'plate'


# ---- reading back and converting ----

def test_read_csv_round_trip(tmp_path):
    rows = make_camera_rows(3, 30, [0, 1], {(1, 0): FOUND, (2, 1): NO_DEPTH})
    write_camera_csv(rows, tmp_path / 'm.csv')
    back = read_csv(tmp_path / 'm.csv')
    assert [r['detected'] for r in back] == [r['detected'] for r in rows]
    assert all(isinstance(r['marker_id'], int) for r in back)
    assert back[2]['dist_m'] == pytest.approx(0.5)


def test_convert_with_world_json(tmp_path):
    write_camera_csv(make_camera_rows(1, 30, [0], {(0, 0): FOUND}), tmp_path / 'cam.csv')
    frame.save(tmp_path / 'world.json', SHIFT, 'plate')
    convert(tmp_path / 'cam.csv', tmp_path / 'world.csv', tmp_path / 'world.json')
    (row,) = read_csv(tmp_path / 'world.csv')
    assert (row['x_m'], row['y_m'], row['z_m']) == pytest.approx((1.3, 2.0, 3.4))
    assert json.loads((tmp_path / 'world.json').read_text())['source'] == 'plate'


def test_convert_without_world_json_is_identity(tmp_path):
    write_camera_csv(make_camera_rows(1, 30, [0], {(0, 0): FOUND}), tmp_path / 'cam.csv')
    convert(tmp_path / 'cam.csv', tmp_path / 'world.csv')
    (row,) = read_csv(tmp_path / 'world.csv')
    assert (row['x_m'], row['z_m']) == (0.3, 0.4)
    _, source = frame.load(tmp_path / 'world.json')
    assert source == 'identity'


# ---- committed examples ----

def test_committed_camera_example_follows_contract():
    lines = read(CAMERA_EXAMPLE)
    assert lines[0] == CAMERA_COLUMNS
    body = lines[1:]
    assert len(body) == len({r[0] for r in body}) * len({r[2] for r in body})
    for r in body:
        detected, valid, z, dist = int(r[3]), int(r[4]), float(r[9]), float(r[10])
        assert valid <= detected
        assert (z > 0) == bool(valid)
        assert (dist > 0) == bool(valid)


def test_committed_world_example_matches_camera_example():
    camera = read_csv(CAMERA_EXAMPLE)
    world = read_csv(WORLD_EXAMPLE)
    assert read(WORLD_EXAMPLE)[0] == WORLD_COLUMNS
    assert len(world) == len(camera)
    for c, w in zip(camera, world):
        assert w['occluded'] == 1 - c['detected']
        assert w['valid'] == c['valid']
        assert (w['x_m'], w['y_m'], w['z_m']) == (c['x_cam_m'], c['y_cam_m'], c['z_cam_m'])  # identity
