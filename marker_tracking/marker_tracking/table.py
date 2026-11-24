"""Assemble the fixed frame x marker tables and write the two CSVs.

markers_camera.csv: what the camera measured, in the camera frame.
    One row per frame and marker, every marker in every frame. Values that are
    not available are 0, and the 0/1 flags say why:
        detected = 0             marker not found: all numeric fields 0
        detected = 1, valid = 0  found, but too little depth: u, v kept, xyz, dist = 0
        detected = 1, valid = 1  everything filled in
    dist_m = straight-line distance from the camera = |(x, y, z)|.

markers_world.csv: the input for SKEL, in the global frame.
    Same rows, positions transformed with T_world_cam (see frame.py).
        occluded = 1   marker not seen in this frame (= 1 - detected)
        valid = 1      x, y, z can be trusted; if valid = 0, x, y, z = 0
    A markers_world.json next to it records T_world_cam and its source.

markers_extra.csv: tracker-specific extras (see the tracker's module), kept apart
    so the two tables above stay identical for every tracker.
    One row per kept detection: frame, time_s, marker_id, then the tracker's
    extra_columns. A value the tracker did not give for a detection is left empty.
"""
import csv
import math
from pathlib import Path

import numpy as np

from marker_tracking import frame as frame_tf

CAMERA_COLUMNS = ['frame', 'time_s', 'marker_id', 'detected', 'valid',
                  'u_px', 'v_px', 'x_cam_m', 'y_cam_m', 'z_cam_m', 'dist_m', 'n_depth_px']
WORLD_COLUMNS = ['frame', 'time_s', 'marker_id', 'x_m', 'y_m', 'z_m', 'occluded', 'valid']
EXTRA_KEY_COLUMNS = ['frame', 'time_s', 'marker_id']  # followed by the tracker's extra_columns

_INT_COLUMNS = {'frame', 'marker_id', 'detected', 'valid', 'occluded', 'n_depth_px'}


def make_camera_rows(n_frames, fps, marker_ids, results):
    """Build all camera-frame rows, sorted by frame, then marker id.

    results: {(frame, marker_id): {'u_px', 'v_px', 'x_cam_m', 'y_cam_m', 'z_cam_m', 'n_depth_px', 'valid'}}
             one entry per detected marker; missing keys = not detected.
    """
    rows = []
    for frame in range(n_frames):
        for marker_id in sorted(marker_ids):
            row = {'frame': frame, 'time_s': frame / fps, 'marker_id': marker_id,
                   'detected': 0, 'valid': 0, 'u_px': 0.0, 'v_px': 0.0,
                   'x_cam_m': 0.0, 'y_cam_m': 0.0, 'z_cam_m': 0.0, 'dist_m': 0.0, 'n_depth_px': 0}
            r = results.get((frame, marker_id))
            if r is not None:
                row['detected'] = 1
                row['u_px'], row['v_px'] = r['u_px'], r['v_px']
                row['n_depth_px'] = r['n_depth_px']
                if r['valid']:
                    row['valid'] = 1
                    xyz = (r['x_cam_m'], r['y_cam_m'], r['z_cam_m'])
                    row['x_cam_m'], row['y_cam_m'], row['z_cam_m'] = xyz
                    row['dist_m'] = math.sqrt(sum(c * c for c in xyz))
            rows.append(row)
    return rows


def make_world_rows(camera_rows, T_world_cam):
    """Turn camera-frame rows into global-frame rows (same order, same count)."""
    xyz_cam = np.array([[r['x_cam_m'], r['y_cam_m'], r['z_cam_m']] for r in camera_rows], dtype=float)
    xyz_world = frame_tf.apply(T_world_cam, xyz_cam.reshape(-1, 3))
    rows = []
    for r, xyz in zip(camera_rows, xyz_world):
        valid = int(r['valid'])
        x, y, z = (float(c) for c in xyz) if valid else (0.0, 0.0, 0.0)
        rows.append({'frame': r['frame'], 'time_s': r['time_s'], 'marker_id': r['marker_id'],
                     'x_m': x, 'y_m': y, 'z_m': z,
                     'occluded': 1 - int(r['detected']), 'valid': valid})
    return rows


def make_extra_rows(fps, extras):
    """Build the extras rows, sorted by frame, then marker id.

    extras: {(frame, marker_id): {column: value}}, the .extra of each kept detection.
    """
    return [{'frame': frame, 'time_s': frame / fps, 'marker_id': marker_id, **extras[(frame, marker_id)]}
            for frame, marker_id in sorted(extras)]


def write_camera_csv(rows, path):
    """Write camera rows with fixed decimals: time 1 us, pixels 0.001 px, xyz/dist 1 um."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CAMERA_COLUMNS)
        for r in rows:
            w.writerow([r['frame'], f"{r['time_s']:.6f}", r['marker_id'], r['detected'], r['valid'],
                        f"{r['u_px']:.3f}", f"{r['v_px']:.3f}",
                        f"{r['x_cam_m']:.6f}", f"{r['y_cam_m']:.6f}", f"{r['z_cam_m']:.6f}",
                        f"{r['dist_m']:.6f}", r['n_depth_px']])


def write_world_csv(rows, path, T_world_cam, source):
    """Write world rows (time 1 us, xyz 1 um) plus <same name>.json with T_world_cam and source."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(WORLD_COLUMNS)
        for r in rows:
            w.writerow([r['frame'], f"{r['time_s']:.6f}", r['marker_id'],
                        f"{r['x_m']:.6f}", f"{r['y_m']:.6f}", f"{r['z_m']:.6f}",
                        r['occluded'], r['valid']])
    frame_tf.save(Path(path).with_suffix('.json'), T_world_cam, source)


def write_extra_csv(rows, extra_columns, path):
    """Write extras rows: time 1 us, extras with 6 decimals (1 um, 1 urad, 1e-6 px), missing values empty."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(EXTRA_KEY_COLUMNS + list(extra_columns))
        for r in rows:
            values = ['' if r.get(c) is None else f'{r[c]:.6f}' for c in extra_columns]
            w.writerow([r['frame'], f"{r['time_s']:.6f}", r['marker_id']] + values)


def read_csv(path):
    """Read any of the three CSVs back into rows: ints for ids and flags, floats for the rest (empty -> nan)."""
    def convert(key, value):
        if key in _INT_COLUMNS:
            return int(value)
        return float(value) if value != '' else math.nan

    with open(path, newline='') as f:
        return [{k: convert(k, v) for k, v in r.items()} for r in csv.DictReader(f)]
