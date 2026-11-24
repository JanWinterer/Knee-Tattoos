"""Run a tracker on one take: take folder in, marker positions out.

    python -m marker_tracking.run --take <take folder> --out <folder>
    python -m marker_tracking.run --take <take folder> --out <folder> --tracker aruco_4x4_50
    python -m marker_tracking.run --take <take folder> --out <folder> --set refinement=none
    python -m marker_tracking.run --list                      # available trackers

The tracker comes from --tracker, otherwise from the take's markers.json.
Every tracker goes through the same steps:
    1. detect   tracker finds markers in each colour frame: id, (u, v), extras   detect.py, trackers/
    2. lift     depth around (u, v) -> xyz in the camera frame                   lift.py
    3. tables   one row per frame and marker; camera -> global frame             table.py, frame.py

Writes to <out>/:
    markers_camera.csv    camera frame          (same columns for every tracker, see table.py)
    markers_world.csv     global frame          (+ markers_world.json with T_world_cam)
    markers_extra.csv     tracker-specific extras, only if the tracker has any
    run.json              which tracker, settings and take produced these files
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from marker_tracking import detect, io, lift, table
from marker_tracking.trackers import get_tracker, list_trackers

CAMERA_CSV = 'markers_camera.csv'
WORLD_CSV = 'markers_world.csv'
EXTRA_CSV = 'markers_extra.csv'
RUN_JSON = 'run.json'


@dataclass
class TrackResult:
    """Everything one tracker run produced, before it is written to files."""
    marker_ids: list                  # ids that get a row in every frame
    ids_from: str                     # 'markers.json' or 'detections'
    camera_rows: list                 # see table.make_camera_rows
    extra_rows: list                  # see table.make_extra_rows
    n_dropped_duplicates: int = 0     # (frame, id) pairs dropped because the id appeared twice
    ignored_ids: list = field(default_factory=list)  # detected ids that are not in markers.json


def markers_for(take, tracker_name):
    """The take's markers.json, if it was written for this tracker; otherwise None.

    Ids and sizes belong to one marker system, so they are not used for another tracker.
    """
    if take.markers is not None and take.markers['tracker'] == tracker_name:
        return take.markers
    return None


def track(take, tracker, markers=None):
    """Run detect -> lift over the whole take and build the table rows. Returns a TrackResult."""
    listed_ids = markers['ids'] if markers else None
    lifted = {}    # (frame, marker_id) -> lift result
    extras = {}    # (frame, marker_id) -> Detection.extra
    n_dropped, ignored = 0, set()

    for frame, detections, dropped_ids in detect.detect_take(take, tracker, markers):
        n_dropped += len(dropped_ids)
        for d in detections:
            if listed_ids is not None and d.marker_id not in listed_ids:
                ignored.add(d.marker_id)  # not on the body: a false detection or an incomplete markers.json
                continue
            key = (frame.index, d.marker_id)
            lifted[key] = lift.lift(frame.depth_m, d.u_px, d.v_px, take.camera)
            if d.extra:
                extras[key] = d.extra

    if listed_ids is not None:
        marker_ids, ids_from = listed_ids, 'markers.json'
    else:
        marker_ids, ids_from = sorted({marker_id for _, marker_id in lifted}), 'detections'

    return TrackResult(
        marker_ids=marker_ids,
        ids_from=ids_from,
        camera_rows=table.make_camera_rows(take.n_frames, take.camera.fps, marker_ids, lifted),
        extra_rows=table.make_extra_rows(take.camera.fps, extras),
        n_dropped_duplicates=n_dropped,
        ignored_ids=sorted(ignored),
    )


def write_outputs(result, take, tracker, markers, out):
    """Write the CSVs and run.json for one tracker run into the folder `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    table.write_camera_csv(result.camera_rows, out / CAMERA_CSV)
    world_rows = table.make_world_rows(result.camera_rows, take.T_world_cam)
    table.write_world_csv(world_rows, out / WORLD_CSV, take.T_world_cam, take.world_source)

    extra_path = out / EXTRA_CSV
    if tracker.extra_columns:
        table.write_extra_csv(result.extra_rows, tracker.extra_columns, extra_path)
    elif extra_path.exists():
        extra_path.unlink()  # left over from an earlier run with another tracker: would be misleading

    run_info = {
        'tracker': tracker.name,
        'settings': tracker.settings(),
        'take': str(Path(take.folder).resolve()),
        'n_frames': take.n_frames,
        'marker_ids': result.marker_ids,
        'ids_from': result.ids_from,
        'marker_size_m': markers['size_m'] if markers else None,
        'lift': {'radius_px': lift.RADIUS_PX, 'min_depth_px': lift.MIN_DEPTH_PX},
        'world_source': take.world_source,
        'n_dropped_duplicates': result.n_dropped_duplicates,
        'ignored_ids': result.ignored_ids,
    }
    (out / RUN_JSON).write_text(json.dumps(run_info, indent=2) + '\n')


def run(take_folder, out_folder, tracker_name=None, **settings):
    """Load a take, run the tracker, write all outputs.

    Returns (take, tracker, markers, result): the loaded io.Take, the Tracker used,
    the markers.json dict that applied (or None) and the TrackResult.
    """
    take = io.load_take(take_folder)
    if tracker_name is None:
        if take.markers is None:
            raise ValueError(f'{take_folder}: no markers.json, so give the tracker explicitly (see --list)')
        tracker_name = take.markers['tracker']
    tracker = get_tracker(tracker_name, **settings)
    markers = markers_for(take, tracker_name)

    result = track(take, tracker, markers)
    write_outputs(result, take, tracker, markers, out_folder)
    return take, tracker, markers, result


def parse_settings(pairs):
    """['refinement=none', 'threshold=0.7'] -> {'refinement': 'none', 'threshold': 0.7}.

    Values are read as JSON where possible (numbers, true/false), otherwise kept as text.
    """
    settings = {}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep or not key:
            raise ValueError(f'--set expects KEY=VALUE, got {pair!r}')
        try:
            settings[key] = json.loads(value)
        except json.JSONDecodeError:
            settings[key] = value
    return settings


def summary(take, tracker, markers, result, out):
    """Short human-readable report of one run (arguments as returned by run())."""
    rows = result.camera_rows
    n_detected = sum(r['detected'] for r in rows)
    n_valid = sum(r['valid'] for r in rows)
    lines = []
    if take.markers is not None and markers is None:
        lines.append(f"note: markers.json is for {take.markers['tracker']}, not {tracker.name}; "
                     f'its ids and size are not used')
    lines += [f'tracker {tracker.name}: ids {result.marker_ids} (from {result.ids_from})',
             f'detected {n_detected} of {len(rows)} frame-marker slots, {n_valid} with a valid 3D position']
    if result.ignored_ids:
        lines.append(f'ignored ids not in markers.json: {result.ignored_ids}')
    if result.n_dropped_duplicates:
        lines.append(f'dropped {result.n_dropped_duplicates} detections whose id appeared twice in a frame')
    lines.append(f'wrote {out}')
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description='Run a marker tracker on one take.')
    p.add_argument('--take', help='take folder (format: see io.py)')
    p.add_argument('--out', help='output folder for the CSVs and run.json')
    p.add_argument('--tracker', help="tracker name (default: from the take's markers.json)")
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='tracker setting, e.g. --set refinement=none (repeatable)')
    p.add_argument('--list', action='store_true', help='list the available trackers and exit')
    args = p.parse_args(argv)

    if args.list:
        print('\n'.join(list_trackers()))
        return
    if not args.take or not args.out:
        p.error('--take and --out are required')

    take, tracker, markers, result = run(args.take, args.out, args.tracker, **parse_settings(args.set))
    print(summary(take, tracker, markers, result, args.out))


if __name__ == '__main__':
    main()
