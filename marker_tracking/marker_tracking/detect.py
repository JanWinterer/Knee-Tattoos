"""Run a tracker over a take and enforce the tracker contract (trackers/base.py).

    for frame, detections, dropped_ids in detect_take(take, tracker, markers):
        frame          io.Frame (index, time_s, color, depth_m)
        detections     checked Detection list, at most one per marker id
        dropped_ids    ids found more than once in this frame (ambiguous, so dropped)

This is the only place that calls tracker.start() and tracker.detect(), so
every tracker is held to the same rules. A detection that breaks them is a bug
in the tracker and raises TrackerError, instead of ending up in a CSV.
"""
import math
import numbers
from collections import Counter

from marker_tracking.trackers import Detection


class TrackerError(Exception):
    """A tracker returned something that breaks the contract in trackers/base.py."""


def check_detection(detection, tracker, width, height):
    """Check one detection against the contract. Returns a copy with plain Python types."""
    who = f'tracker {tracker.name!r}'
    if not isinstance(detection, Detection):
        raise TrackerError(f'{who}: detect() must return Detection objects, got {type(detection).__name__}')

    marker_id = detection.marker_id
    if isinstance(marker_id, bool) or not isinstance(marker_id, numbers.Integral) or marker_id < 0:
        raise TrackerError(f'{who}: marker_id must be an integer >= 0, got {marker_id!r}')

    u, v = float(detection.u_px), float(detection.v_px)
    if not (math.isfinite(u) and math.isfinite(v)):
        raise TrackerError(f'{who}: marker {marker_id} has a non-finite centre ({u}, {v})')
    # Pixel centres run from 0 to width - 1, so the image itself spans -0.5 .. width - 0.5.
    if not (-0.5 <= u <= width - 0.5 and -0.5 <= v <= height - 0.5):
        raise TrackerError(f'{who}: marker {marker_id} centre ({u:.1f}, {v:.1f}) is outside the '
                           f'{width} x {height} image')

    undeclared = sorted(set(detection.extra) - set(tracker.extra_columns))
    if undeclared:
        raise TrackerError(f'{who}: extra values {undeclared} are not listed in extra_columns')
    extra = {}
    for key, value in detection.extra.items():
        try:
            extra[key] = float(value)
        except (TypeError, ValueError):
            raise TrackerError(f'{who}: extra value {key!r} = {value!r} is not a number') from None

    return Detection(int(marker_id), u, v, extra)


def check_detections(detections, tracker, width, height):
    """Check all detections of one frame and drop ids that occur more than once.

    Returns (kept detections, dropped ids).
    """
    if not isinstance(detections, (list, tuple)):
        raise TrackerError(f'tracker {tracker.name!r}: detect() must return a list, '
                           f'got {type(detections).__name__}')
    checked = [check_detection(d, tracker, width, height) for d in detections]
    count = Counter(d.marker_id for d in checked)
    dropped_ids = sorted(marker_id for marker_id, n in count.items() if n > 1)
    kept = [d for d in checked if count[d.marker_id] == 1]
    return kept, dropped_ids


def detect_take(take, tracker, markers=None):
    """Run `tracker` on every frame of `take` in order; yields (frame, detections, dropped_ids).

    markers: the markers.json dict handed to the tracker (or None).
    """
    tracker.start(take.camera, markers)
    for frame in take:
        detections = tracker.detect(frame.color, frame.depth_m)
        kept, dropped_ids = check_detections(detections, tracker, take.camera.width, take.camera.height)
        yield frame, kept, dropped_ids
