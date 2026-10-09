"""Pictures of what the LiDAR sees: range image with coloured segments, top view and (optionally) the camera frame.

  python -m testing.tools.perception.render_scan --demo                     # a synthetic road with a car and a person
  python -m testing.tools.perception.render_scan --recording dev_default_seed12345 --frame 0
  python -m testing.tools.perception.render_scan --recording dev_default_seed12345 --frames 0:100:10 --camera
  python -m testing.tools.perception.render_scan --recording dev_default_seed12345 --frame 5 --ground height

--recording    a folder name inside testing/data/recordings, or a path
--frame N      scan number (index in lidar.bin);  --frames A:B:S  a range with step (B is not included)
--ground       range_image (default) or height: the ground-removal method
--camera       add the camera frame closest in time, with the LiDAR and refined extents; also switches the refinement on
--no-refine    with --camera: show the extents but do not apply the refinement
--out          output folder, default testing/data/scan_views
Every image has the range image on top (each segment in its own colour, closer = lighter and paler, farther = darker;
ground tan, discarded clusters dark red, no return black, max range dark blue) and below it the top view and the
camera panel side by side.
"""
import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

import cv2
import numpy as np

from server.communication import protocol as P
from server.geometry.calibration import Calibration
from server.perception.camera_refinement import RefinementConfig
from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from testing.support import scan_view
from testing.support.dev_recording import camera_path, load_camera_index, load_lidar, load_pose
from testing.support.paths import DATA_DIR, RECORDINGS_DIR


def lidar_message(rec) -> bytes:
    return P.build_message(P.MsgType.LIDAR, int(rec["frame"]), P.lidar_payload(rec["ranges"]), t=float(rec["t"]),
                           pose=tuple(rec["pose"]), speed=float(rec["motion"][0]), yaw_rate=float(rec["motion"][1]),
                           steering_angle=float(rec["motion"][2]))


def camera_for(folder: Path, cam_index, pose_df, t: float, max_gap: float):
    """(header-like with pose and t, rgb image) of the recorded camera frame closest to t, or (None, None)."""
    i = int(np.argmin(np.abs(cam_index["t"].to_numpy() - t)))
    ct = float(cam_index["t"].iloc[i])
    if abs(ct - t) > max_gap:
        return None, None
    bgr = cv2.imread(str(camera_path(folder, int(cam_index["frame"].iloc[i]))))
    if bgr is None:
        return None, None
    pose = tuple(float(np.interp(ct, pose_df["t"], pose_df[c])) for c in ("x", "y", "z", "yaw", "pitch", "roll"))
    return SimpleNamespace(t=ct, pose=pose), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def demo(args, out: Path):
    from testing.support.synthetic_camera import make_camera, render as render_camera
    from testing.support.synthetic_scene import Scene, make_lidar
    from server.geometry.calibration import EgoCalibration
    lidar, camera = make_lidar(cols=601), make_camera()
    calib = Calibration("demo", EgoCalibration(2.7, 4.4, 1.8, 1.5, 0.9, 0.8), camera, lidar, 100.0)
    scene = (Scene([(0.0, 0.0), (45.0, 0.05)]).add_box(22, -1.0, 4.5, 1.8, 1.5).add_box(22.5, 1.2, 0.6, 0.5, 1.7)
             .add_box(12, 4.5, 0.6, 0.5, 1.7).add_box(45, -3, 4.5, 1.8, 1.5).add_box(36, 8, 1.0, 1.0, 2.5))
    img, _, geo = scene.render(lidar)
    pipe = make_pipeline(calib, args)
    rgb = render_camera(camera, [(22, -1.0, 4.5, 1.8, 1.5), (22.5, 1.2, 0.6, 0.5, 1.7), (12, 4.5, 0.6, 0.5, 1.7),
                                 (45, -3, 4.5, 1.8, 1.5), (36, 8, 1.0, 1.0, 2.5)])
    result = pipe.process(img, SimpleNamespace(t=img.t, pose=img.pose) if args.camera else None, rgb if args.camera else None)
    write(out / "demo.png", result, calib, geo, rgb if args.camera else None)


def make_pipeline(calib, args):
    ref = RefinementConfig(enabled=args.camera and not args.no_refine)
    cfg = PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground)), refinement=ref)
    pipe = PerceptionPipeline(calib, cfg)
    if args.camera and args.no_refine:                      # extents are still shown: refine, but do not apply
        pipe.refiner.config = replace(pipe.refiner.config, enabled=True)
    return pipe


def make_gif(image_paths: list[Path], output_path: Path, fps: float):
    frames = []

    try:
        for path in image_paths:
            with Image.open(path) as img:
                frames.append(img.convert("RGB"))

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / fps),
            loop=0,
            optimize=False,
        )
        print("Created GIF:", output_path)

    finally:
        for frame in frames:
            frame.close()


def write(path: Path, result, calib, geo, rgb):
    panels = [scan_view.range_panel(result, calib), scan_view.top_view(result, calib, geo)]
    if rgb is not None:
        panels.append(scan_view.camera_panel(result, rgb))
    scan_view.save_rgb(path, scan_view.compose(panels))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--recording")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--frames", default=None, help="A:B:S")
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--out", default=str(DATA_DIR / "scan_views"))
    ap.add_argument("--gif", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        return demo(args, out)
    if not args.recording:
        ap.error("give --recording (or --demo)")
    folder = Path(args.recording)
    if not folder.exists():
        folder = RECORDINGS_DIR / args.recording
    calib = Calibration.from_json(folder / "calibration.json")
    scans = load_lidar(folder)
    if args.frames:
        a, b, s = (int(x) for x in args.frames.split(":"))
        picks = range(a, min(b, len(scans)), s)
    else:
        picks = [args.frame or 0]
    pipe = make_pipeline(calib, args)
    cam_index = load_camera_index(folder) if args.camera else None
    pose_df = load_pose(folder) if args.camera else None
    image_paths = []
    for k in picks:
        rec = scans[k]
        img = pipe.builder.ingest(lidar_message(rec))
        hdr, rgb = camera_for(folder, cam_index, pose_df, img.t, pipe.config.refinement.max_time_gap_s) if args.camera else (None, None)
        result = pipe.process(img, hdr, rgb)
        image_path = out / f"{folder.name}_scan{k:05d}.png"
        write(image_path, result, calib, pipe.builder.geometry, rgb)
        image_paths.append(image_path)
    if args.gif is not None and args.gif > 0 and image_paths:
        make_gif(image_paths, out / f"{folder.name}.gif", args.gif)


if __name__ == "__main__":
    main()