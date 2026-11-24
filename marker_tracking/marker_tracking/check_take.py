"""Check a take folder before tracking it (format: see io.py).

    python -m marker_tracking.check_take <take folder>
    python -m marker_tracking.check_take <take folder> --every 10    # only read every 10th frame

Errors: the tracker cannot run or would give wrong numbers. Warnings: looks
suspicious, check by hand. Exit code 1 if there is any error.
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from marker_tracking import frame as frame_tf
from marker_tracking import io

# Plausibility limits (warnings only)
MIN_VALID_DEPTH_FRACTION = 0.10  # at least 10 % of pixels should have depth
DEPTH_RANGE_M = (0.1, 5.0)       # median depth of a frame should be in this range
MAX_FX_FY_RATIO = 1.1            # fx and fy should be about equal (square pixels)
MAX_PRINCIPAL_OFFSET = 0.15      # cx, cy within 15 % of the image size from the centre
FOV_DEG = (20.0, 120.0)          # horizontal field of view


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors

    def text(self):
        lines = [f'ERROR   {m}' for m in self.errors] + [f'WARNING {m}' for m in self.warnings]
        lines += [f'        {m}' for m in self.info]
        lines.append('OK' if self.ok else f'FAILED ({len(self.errors)} errors)')
        return '\n'.join(lines)


def check_camera(data, r):
    """Check camera.json content. Returns io.Camera or None."""
    try:
        cam = io.Camera.from_dict(data)
    except (ValueError, TypeError) as e:
        r.errors.append(str(e))
        return None
    if cam.width <= 0 or cam.height <= 0:
        r.errors.append(f'image size {cam.width} x {cam.height}')
    if cam.fps <= 0:
        r.errors.append(f'fps = {cam.fps}')
    if cam.fx <= 0 or cam.fy <= 0:
        r.errors.append(f'focal lengths fx = {cam.fx}, fy = {cam.fy} must be > 0')
    if cam.depth_scale_m <= 0:
        r.errors.append(f'depth_scale_m = {cam.depth_scale_m} must be > 0')
    if len(cam.dist_coeffs) not in (4, 5, 8, 12, 14):
        r.errors.append(f'dist_coeffs has {len(cam.dist_coeffs)} values, OpenCV takes 4, 5, 8, 12 or 14')
    if r.errors:
        return None

    if max(cam.fx, cam.fy) / min(cam.fx, cam.fy) > MAX_FX_FY_RATIO:
        r.warnings.append(f'fx = {cam.fx:.1f} and fy = {cam.fy:.1f} differ a lot (non-square pixels?)')
    if not (0 <= cam.cx < cam.width and 0 <= cam.cy < cam.height):
        r.errors.append(f'principal point ({cam.cx:.1f}, {cam.cy:.1f}) outside the image')
    elif (abs(cam.cx - (cam.width - 1) / 2) > MAX_PRINCIPAL_OFFSET * cam.width
          or abs(cam.cy - (cam.height - 1) / 2) > MAX_PRINCIPAL_OFFSET * cam.height):
        r.warnings.append(f'principal point ({cam.cx:.1f}, {cam.cy:.1f}) far from the image centre')
    fov = np.degrees(2 * np.arctan(cam.width / (2 * cam.fx)))
    if not FOV_DEG[0] <= fov <= FOV_DEG[1]:
        r.warnings.append(f'horizontal field of view {fov:.1f} deg (focal length in mm instead of px?)')
    if cam.depth_scale_m > 0.01:
        r.warnings.append(f'depth_scale_m = {cam.depth_scale_m}: coarser than 1 cm per unit')
    max_depth = 65535 * cam.depth_scale_m
    if max_depth < DEPTH_RANGE_M[1]:
        r.warnings.append(f'depth_scale_m = {cam.depth_scale_m}: 16 bit only reaches {max_depth:.2f} m')
    r.info.append(f'camera: {cam.width} x {cam.height} px, {cam.fps:g} fps, horizontal FOV {fov:.1f} deg, '
                  f'depth step {cam.depth_scale_m * 1000:g} mm, source {cam.source}')
    return cam


def check_world(path, r):
    if not path.is_file():
        r.info.append('world.json: none, global frame = camera frame')
        return
    try:
        T, source = frame_tf.load(path)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        r.errors.append(f'world.json: {e}')
        return
    r.info.append(f'world.json: source {source}, camera at {np.round(T[:3, 3], 4).tolist()} m in the global frame')


def check_frame_files(folder, r):
    color = io.frame_indices(folder / io.COLOR_DIR)
    depth = io.frame_indices(folder / io.DEPTH_DIR)
    for name, idx in ((io.COLOR_DIR, color), (io.DEPTH_DIR, depth)):
        if not idx:
            r.errors.append(f'{name}/: no frames (expected {name}/000000.png, ...)')
        elif idx != list(range(len(idx))):
            gaps = sorted(set(range(idx[-1] + 1)) - set(idx))
            r.errors.append(f'{name}/: frame numbers must be 0, 1, 2, ... without gaps; missing {gaps[:10]}')
    only_color, only_depth = sorted(set(color) - set(depth)), sorted(set(depth) - set(color))
    if only_color:
        r.errors.append(f'frames without depth: {only_color[:10]}')
    if only_depth:
        r.errors.append(f'frames without colour: {only_depth[:10]}')
    return sorted(set(color) & set(depth))


def check_frame(folder, index, cam, r):
    """Check one frame pair. Returns (valid depth fraction, median depth m) or None."""
    name = io.frame_name(index)
    color = cv2.imread(str(folder / io.COLOR_DIR / name), cv2.IMREAD_UNCHANGED)
    raw = cv2.imread(str(folder / io.DEPTH_DIR / name), cv2.IMREAD_UNCHANGED)
    if color is None:
        r.errors.append(f'frame {index}: cannot read colour image')
    elif color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        r.errors.append(f'frame {index}: colour must be 8-bit 3-channel, got {color.dtype} {color.shape}')
    elif color.shape[:2] != (cam.height, cam.width):
        r.errors.append(f'frame {index}: colour is {color.shape[1]} x {color.shape[0]}, '
                        f'camera.json says {cam.width} x {cam.height}')
    if raw is None:
        r.errors.append(f'frame {index}: cannot read depth image')
        return None
    if raw.dtype != np.uint16 or raw.ndim != 2:
        r.errors.append(f'frame {index}: depth must be 16-bit single channel, got {raw.dtype} {raw.shape}')
        return None
    if raw.shape != (cam.height, cam.width):
        r.errors.append(f'frame {index}: depth is {raw.shape[1]} x {raw.shape[0]}, '
                        f'camera.json says {cam.width} x {cam.height}')
        return None
    good = raw > 0
    fraction = float(good.mean())
    median = float(np.median(raw[good])) * cam.depth_scale_m if good.any() else 0.0
    return fraction, median


def check_take(folder, every=1):
    """Check everything; returns a Report."""
    folder = Path(folder)
    r = Report()
    camera_path = folder / io.CAMERA_FILE
    if not camera_path.is_file():
        r.errors.append(f'{camera_path} not found (is this a take folder?)')
        return r
    try:
        data = json.loads(camera_path.read_text())
    except json.JSONDecodeError as e:
        r.errors.append(f'camera.json is not valid JSON: {e}')
        return r
    cam = check_camera(data, r)
    check_world(folder / io.WORLD_FILE, r)
    indices = check_frame_files(folder, r)
    if cam is None or not indices:
        return r

    stats = {}
    for i in indices[::max(1, every)]:
        s = check_frame(folder, i, cam, r)
        if s is not None:
            stats[i] = s
    if not stats:
        return r

    sparse = [i for i, (frac, _) in stats.items() if frac < MIN_VALID_DEPTH_FRACTION]
    if sparse:
        r.warnings.append(f'{len(sparse)} frames with depth on < {MIN_VALID_DEPTH_FRACTION:.0%} of pixels, '
                          f'e.g. {sparse[:5]}')
    off = [i for i, (frac, med) in stats.items() if frac > 0 and not DEPTH_RANGE_M[0] <= med <= DEPTH_RANGE_M[1]]
    if off:
        meds = [stats[i][1] for i in off[:3]]
        r.warnings.append(f'{len(off)} frames with median depth outside {DEPTH_RANGE_M} m, e.g. '
                          f'{[round(m, 4) for m in meds]} m (wrong depth_scale_m?)')

    fracs = [f for f, _ in stats.values()]
    meds = [m for f, m in stats.values() if f > 0]
    r.info.append(f'frames: {len(indices)} ({len(indices) / cam.fps:.2f} s), checked {len(stats)}')
    r.info.append(f'depth coverage: {min(fracs):.0%} .. {max(fracs):.0%} of pixels')
    if meds:
        r.info.append(f'median depth: {min(meds):.3f} .. {max(meds):.3f} m')
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description='Check a take folder (format: see marker_tracking/io.py).')
    p.add_argument('take')
    p.add_argument('--every', type=int, default=1, help='only read every n-th frame (default: all)')
    args = p.parse_args(argv)
    report = check_take(args.take, args.every)
    print(report.text())
    raise SystemExit(0 if report.ok else 1)


if __name__ == '__main__':
    main()
