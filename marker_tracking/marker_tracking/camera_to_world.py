"""Convert markers_camera.csv to markers_world.csv (+ markers_world.json).

    python -m marker_tracking.camera_to_world markers_camera.csv markers_world.csv
    python -m marker_tracking.camera_to_world markers_camera.csv markers_world.csv --world world.json

Without --world the global frame is the camera frame (identity).
"""
import argparse

from marker_tracking import frame, table


def convert(camera_csv, world_csv, world_json=None):
    """Read camera rows, transform them, write the world CSV. Returns the world rows."""
    T, source = frame.load(world_json)
    rows = table.make_world_rows(table.read_csv(camera_csv), T)
    table.write_world_csv(rows, world_csv, T, source)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('camera_csv')
    p.add_argument('world_csv')
    p.add_argument('--world', default=None, help='JSON with T_world_cam (default: identity)')
    args = p.parse_args(argv)
    rows = convert(args.camera_csv, args.world_csv, args.world)
    print(f'wrote {args.world_csv} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
