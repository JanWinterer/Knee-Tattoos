"""Read and write takes: one recording = colour + depth frames + camera description.

Take folder format (version 1)
------------------------------
Written by the Blender exporter, the D456 recorder or a test generator; read by the tracker.

    <take>/
        camera.json          intrinsics, image size, fps, depth units   (required)
        world.json           T_world_cam, see frame.py                  (optional, missing = identity)
        markers.json         which markers are on the body               (optional, see below)
        color/000000.png     8-bit colour, 3 channels                   (one per frame)
        depth/000000.png     16-bit grey, depth aligned to color         (one per frame)

camera.json:
    {
      "format_version": 1,
      "width": 1280, "height": 720,     # pixels, same for color and depth
      "fps": 30,                        # time_s = frame / fps
      "fx": 645.49, "fy": 644.72,       # focal lengths [px]
      "cx": 641.45, "cy": 371.51,       # principal point [px], origin = centre of the top-left pixel
      "depth_scale_m": 0.001,           # metres per depth unit: depth_m = png value * depth_scale_m
      "dist_coeffs": [0, 0, 0, 0, 0],   # optional, OpenCV order k1 k2 p1 p2 k3, default all 0
      "source": "blender"               # optional, free text: blender, d456, synthetic, ...
    }

markers.json:
    {
      "tracker": "aruco_4x4_50",        # tracker that reads these markers, see trackers.list_trackers()
      "ids": [0, 1, 2, 3],              # every marker on the body; each gets a row in every frame
      "size_m": 0.03                    # optional, outer side of the marker [m] (for tracker extras)
    }
    Without it, the tracker has to be named on the command line and the
    rows cover the ids that were detected somewhere in the take.

Conventions:
    - Frames are numbered 000000, 000001, ... with no gaps; color/ and depth/ have the same numbers.
    - Camera frame = OpenCV: x right, y down, z forward (out of the lens), metres.
    - Depth = z (distance along the optical axis), not the distance along the ray.
    - Depth pixel (u, v) belongs to colour pixel (u, v) (aligned). Depth value 0 = no depth.

Use:
    take = load_take('takes/knee_01')
    take.camera.K, take.T_world_cam, take.world_source, take.n_frames
    take.markers                                # markers.json as a dict, or None
    for f in take:                              # or take.frame(i)
        f.index, f.time_s, f.color, f.depth_m   # color BGR uint8 (H, W, 3); depth_m float32 (H, W), 0 = none

    write_take_camera(folder, camera)           # create a take (for generators and recorders)
    write_take_frame(folder, i, color, depth_m, camera.depth_scale_m)
    write_take_markers(folder, 'aruco_4x4_50', [0, 1, 2, 3], size_m=0.03)
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from marker_tracking import frame as frame_tf

FORMAT_VERSION = 1
CAMERA_FILE = 'camera.json'
WORLD_FILE = 'world.json'
MARKERS_FILE = 'markers.json'
COLOR_DIR = 'color'
DEPTH_DIR = 'depth'
COLOR_EXTS = ('.png', '.jpg', '.jpeg')  # depth stays .png only (16 bit, lossless)
REQUIRED_CAMERA_KEYS = ['format_version', 'width', 'height', 'fps', 'fx', 'fy', 'cx', 'cy', 'depth_scale_m']


@dataclass
class Camera:
    width: int
    height: int
    fps: float
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float
    dist_coeffs: list = field(default_factory=lambda: [0.0] * 5)
    source: str = 'unknown'

    @property
    def K(self):
        """3x3 intrinsic matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)

    @classmethod
    def from_dict(cls, data):
        missing = [k for k in REQUIRED_CAMERA_KEYS if k not in data]
        if missing:
            raise ValueError(f'{CAMERA_FILE}: missing keys {missing}')
        if data['format_version'] != FORMAT_VERSION:
            raise ValueError(f"{CAMERA_FILE}: format_version {data['format_version']}, "
                             f'this code reads version {FORMAT_VERSION}')
        return cls(width=int(data['width']), height=int(data['height']), fps=float(data['fps']),
                   fx=float(data['fx']), fy=float(data['fy']), cx=float(data['cx']), cy=float(data['cy']),
                   depth_scale_m=float(data['depth_scale_m']),
                   dist_coeffs=[float(c) for c in data.get('dist_coeffs', [0.0] * 5)],
                   source=str(data.get('source', 'unknown')))

    def to_dict(self):
        return {'format_version': FORMAT_VERSION, **asdict(self)}


@dataclass
class Frame:
    index: int
    time_s: float
    color: np.ndarray    # BGR uint8 (H, W, 3)
    depth_m: np.ndarray  # float32 (H, W) metres, 0 = no depth


def frame_name(index):
    return f'{index:06d}.png'


def frame_indices(folder, exts=('.png',)):
    """Sorted frame numbers found in a color/ or depth/ folder (files named 000123.png)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(int(p.stem) for p in folder.iterdir() if p.suffix.lower() in exts and p.stem.isdigit())


def frame_path(folder, index, exts=('.png',)):
    """Path of frame <index> in folder: the first existing extension of exts."""
    for ext in exts:
        path = Path(folder) / frame_name(index).replace('.png', ext)
        if path.is_file():
            return path
    return Path(folder) / frame_name(index)


def read_color(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f'cannot read colour image {path}')
    return image


def read_depth(path, depth_scale_m):
    """16-bit depth png -> float32 metres (0 stays 0 = no depth)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f'cannot read depth image {path}')
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError(f'{path}: depth must be 16-bit single channel, got {raw.dtype} {raw.shape}')
    return raw.astype(np.float32) * np.float32(depth_scale_m)


class Take:
    def __init__(self, folder):
        self.folder = Path(folder)
        camera_path = self.folder / CAMERA_FILE
        if not camera_path.is_file():
            raise FileNotFoundError(f'{camera_path} (is {self.folder} a take folder?)')
        self.camera = Camera.from_dict(json.loads(camera_path.read_text()))

        world_path = self.folder / WORLD_FILE
        self.T_world_cam, self.world_source = frame_tf.load(world_path if world_path.is_file() else None)

        markers_path = self.folder / MARKERS_FILE
        self.markers = read_markers(markers_path) if markers_path.is_file() else None

        color = frame_indices(self.folder / COLOR_DIR, COLOR_EXTS)
        depth = frame_indices(self.folder / DEPTH_DIR)
        if color != depth:
            raise ValueError(f'{self.folder}: color/ and depth/ have different frame numbers '
                             f'({len(color)} vs {len(depth)} files)')
        if color != list(range(len(color))):
            raise ValueError(f'{self.folder}: frames must be numbered 000000, 000001, ... without gaps')
        self.n_frames = len(color)

    def frame(self, index):
        if not 0 <= index < self.n_frames:
            raise IndexError(f'frame {index} not in 0..{self.n_frames - 1}')
        name = frame_name(index)
        return Frame(index=index, time_s=index / self.camera.fps,
                     color=read_color(frame_path(self.folder / COLOR_DIR, index, COLOR_EXTS)),
                     depth_m=read_depth(self.folder / DEPTH_DIR / name, self.camera.depth_scale_m))

    def __len__(self):
        return self.n_frames

    def __iter__(self):
        for i in range(self.n_frames):
            yield self.frame(i)


def load_take(folder):
    return Take(folder)


def read_markers(path):
    """Read and check markers.json. Returns {'tracker': str, 'ids': sorted [int], 'size_m': float or None}."""
    data = json.loads(Path(path).read_text())
    tracker = data.get('tracker')
    if not isinstance(tracker, str) or not tracker:
        raise ValueError(f"{path}: 'tracker' must be a tracker name")
    ids = data.get('ids')
    if (not isinstance(ids, list) or not ids
            or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids)):
        raise ValueError(f"{path}: 'ids' must be a non-empty list of integers >= 0")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: 'ids' contains duplicates")
    size_m = data.get('size_m')
    if size_m is not None and not (isinstance(size_m, (int, float)) and size_m > 0):
        raise ValueError(f"{path}: 'size_m' must be a positive number of metres")
    return {'tracker': tracker, 'ids': sorted(ids), 'size_m': None if size_m is None else float(size_m)}


def write_take_camera(folder, camera, T_world_cam=None, world_source='identity'):
    """Create the take folder, camera.json and (if T_world_cam is given) world.json."""
    folder = Path(folder)
    (folder / COLOR_DIR).mkdir(parents=True, exist_ok=True)
    (folder / DEPTH_DIR).mkdir(parents=True, exist_ok=True)
    (folder / CAMERA_FILE).write_text(json.dumps(camera.to_dict(), indent=2) + '\n')
    if T_world_cam is not None:
        frame_tf.save(folder / WORLD_FILE, T_world_cam, world_source)


def write_take_markers(folder, tracker, ids, size_m=None):
    """Write markers.json: the tracker for these markers, the ids on the body and (optionally) their size [m]."""
    data = {'tracker': tracker, 'ids': sorted(int(i) for i in ids)}
    if size_m is not None:
        data['size_m'] = float(size_m)
    (Path(folder) / MARKERS_FILE).write_text(json.dumps(data, indent=2) + '\n')


def depth_to_png_values(depth_m, depth_scale_m):
    """float metres -> uint16 units. NaN, inf and <= 0 become 0 (no depth); too far is an error."""
    d = np.asarray(depth_m, dtype=np.float64)
    good = np.isfinite(d) & (d > 0)
    units = np.zeros(d.shape, dtype=np.float64)
    units[good] = np.round(d[good] / depth_scale_m)
    if units.max(initial=0) > 65535:
        raise ValueError(f'depth {d[good].max():.3f} m does not fit 16 bit at depth_scale_m={depth_scale_m}')
    return units.astype(np.uint16)


def write_take_frame(folder, index, color, depth_m, depth_scale_m):
    """Write color/<index>.png (BGR uint8) and depth/<index>.png (16-bit)."""
    folder = Path(folder)
    color = np.asarray(color)
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f'color must be uint8 (H, W, 3), got {color.dtype} {color.shape}')
    if color.shape[:2] != np.shape(depth_m):
        raise ValueError(f'color {color.shape[:2]} and depth {np.shape(depth_m)} differ in size')
    name = frame_name(index)
    if not cv2.imwrite(str(folder / COLOR_DIR / name), color):
        raise OSError(f'cannot write {folder / COLOR_DIR / name}')
    if not cv2.imwrite(str(folder / DEPTH_DIR / name), depth_to_png_values(depth_m, depth_scale_m)):
        raise OSError(f'cannot write {folder / DEPTH_DIR / name}')
