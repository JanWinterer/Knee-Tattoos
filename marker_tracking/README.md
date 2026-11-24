# Marker tracking

Video of a body with markers in, marker positions (metres) out.
Work in progress.

## Input: take folder

One recording = one folder. Written by the Blender exporter or the D456 recorder, read by the tracker.
Full specification in `marker_tracking/io.py`.

    <take>/
        camera.json          fx, fy, cx, cy [px], width, height, fps, depth_scale_m
        world.json           optional T_world_cam (missing = global frame is the camera frame)
        color/000000.png     8-bit colour
        depth/000000.png     16-bit, aligned to colour, depth_m = value * depth_scale_m, 0 = no depth

Depth is z along the optical axis (OpenCV camera: x right, y down, z forward). Check a take with

```bash
python -m marker_tracking.check_take <take folder>
```

## Outputs

One row per frame and marker, every marker in every frame. `time_s` = `frame / fps`.

`markers_camera.csv`: what the camera measured, camera frame

    frame,time_s,marker_id,detected,valid,u_px,v_px,x_cam_m,y_cam_m,z_cam_m,dist_m,n_depth_px

`markers_world.csv`: input for SKEL, global frame

    frame,time_s,marker_id,x_m,y_m,z_m,occluded,valid

- `occluded = 1`: marker not seen in this frame. `valid = 1`: xyz can be trusted, otherwise xyz = 0.
- `markers_world.json` next to it holds the `T_world_cam` used and its `source`.
  For now `identity` (global frame = camera frame); later the calibration plate.

Convert one into the other (`--world` is optional, default identity):

```bash
python -m marker_tracking.camera_to_world markers_camera.csv markers_world.csv --world world.json
```

Example files in `examples/` (regenerate with `python examples/make_example.py`).

## Layout

- `marker_tracking/`: the package, one step per file (`io`, `detect`, `lift`, `frame`, `table`, `camera_to_world`, `evaluate`, `run`)
- `blender/export_take.py`: renders a take from the Blender knee scene (virtual ground truth)
- `tests/`: pytest tests

## Tests

From this folder:

```bash
pytest
```
