"""Pictures of what the LiDAR sees: range image with coloured segments, top view and (optionally) the camera frame.

  python -m testing.tools.perception.render_scan --demo                     # a synthetic road with a car and a person
  python -m testing.tools.perception.render_scan --recording dev_default_seed12345 --frame 0
  python -m testing.tools.perception.render_scan --recording dev_default_seed12345 --frames 0:100:10 --camera
  python -m testing.tools.perception.render_scan --recording dev_default_seed12345 --frame 5 --ground height
  python -m testing.tools.perception.render_scan --recording dev_default_seed12345 --frames 100:140:1 --tracks --truth --focus 5 --gif 5

--recording    a folder name inside testing/data/recordings, or a path
--frame N      scan number (index in lidar.bin);  --frames A:B:S  a range with step (B is not included)
--ground       range_image (default) or height: the ground-removal method
--camera       add the camera frame closest in time, with the LiDAR and refined extents; also switches the refinement on
--no-refine    with --camera: show the extents but do not apply the refinement
--truth        draw the ground-truth footprints and centres of the hazards in the top view (needs the ground-truth files)
--view-range   depth of the top view in metres (default 50)
--focus ID     with --truth: top view zoomed on this hazard (id as in hazards.csv), a square of 2 x --focus-radius metres
--focus-radius half side of the zoomed window in metres (default 8)
--tracks       (step 6.11) run the tracker over the scans from the first one up to the last picked and draw it on the top
               view: tracks (gold = confirmed, grey = tentative, red = coasting; a square marker = flagged static), velocity
               arrow, 95 % position ellipse, the extent the track remembers as a box, a white line from every cluster to its
               track, and the births, merges and deletions of the scan; with the camera bearing updates (their search regions
               and the edges found) in the camera panel
--lidar-only   with --tracks: no camera bearing updates and no camera panel
--gif FPS      also make a gif of the written images
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

from server.communication import protocol as P
from server.geometry.calibration import Calibration
from server.perception.camera_refinement import RefinementConfig
from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from testing.support import scan_view, track_view
from testing.support.dev_recording import load_lidar
from testing.support.ground_truth import GroundTruth
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import CameraFrames, lidar_message, resolve


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


def write(path: Path, result, calib, geo, rgb, truths=None, view_range_m: float = 50.0, focus=None):
    panels = [scan_view.range_panel(result, calib), scan_view.top_view(result, calib, geo, view_range_m=view_range_m, truths=truths, focus=focus)]
    if rgb is not None:
        panels.append(scan_view.camera_panel(result, rgb))
    scan_view.save_rgb(path, scan_view.compose(panels))
    print("wrote", path)


def render_with_tracks(args, folder, calib, scans, picks, truth, out: Path) -> list:
    """The tracker runs over every scan from the first to the last picked (it needs the history); pictures only of the picked ones."""
    from server.tracking.measurement import clusters_to_world
    from server.tracking.tracker import Tracker
    from testing.support.camera_roi_view import draw_rois
    from testing.support.fusion_study import camera_timeline, load_rgb
    pipe = make_pipeline(calib, args)
    tracker = Tracker()
    sensor_xy = pipe.builder.geometry.origin[:2]
    cams = [] if args.lidar_only else camera_timeline(folder)
    ci, last_cam, wanted, paths = 0, None, set(picks), []
    for k in range(0, max(picks) + 1):
        img = pipe.builder.ingest(lidar_message(scans[k]))
        while ci < len(cams) and cams[ci][0] < img.t:                       # camera frames captured before this scan, in time order
            ct, fr, pose = cams[ci]
            rgb = load_rgb(folder, fr)
            if rgb is not None:
                diag = []
                last_cam = (ct, rgb, diag, tracker.step_camera(rgb, pose, ct, calib.camera, diag))
            ci += 1
        result = pipe.process(img, None, None)
        world = clusters_to_world(result.clusters, img, sensor_xy)
        res = tracker.step(world, img.t)
        if k not in wanted:
            continue
        tracks, links, head = track_view.overlay(tracker, res, world, img.pose)
        objs = truth.objects_at(img.t) if truth else None
        focus = None
        if args.focus is not None and objs:
            hit = [o for o in objs if o.id == args.focus]
            focus = (float(hit[0].centre[0]), float(hit[0].centre[1]), args.focus_radius) if hit else None
        panels = [scan_view.range_panel(result, calib),
                  scan_view.top_view(result, calib, pipe.builder.geometry, view_range_m=args.view_range, truths=objs, focus=focus,
                                     tracks=tracks, links=links, headline=head)]
        if last_cam is not None:
            ct, rgb, diag, results = last_cam
            frame = draw_rois(rgb, diag, results, tracker.config.bearing, ct)
            panels.append(frame[:, :rgb.shape[1]])                          # the frame with the search regions, not the edge map
        path = out / f"{folder.name}_tracks{k:05d}.png"
        scan_view.save_rgb(path, scan_view.compose(panels))
        print("wrote", path)
        paths.append(path)
    return paths


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
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--focus", type=int, default=None)
    ap.add_argument("--focus-radius", type=float, default=8.0)
    ap.add_argument("--view-range", type=float, default=50.0)
    ap.add_argument("--tracks", action="store_true")
    ap.add_argument("--lidar-only", action="store_true")
    ap.add_argument("--gif", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        return demo(args, out)
    if not args.recording:
        ap.error("give --recording (or --demo)")
    folder = resolve(args.recording)
    calib = Calibration.from_json(folder / "calibration.json")
    scans = load_lidar(folder)
    if args.frames:
        a, b, s = (int(x) for x in args.frames.split(":"))
        picks = range(a, min(b, len(scans)), s)
    else:
        picks = [args.frame or 0]
    truth = GroundTruth.load(folder) if args.truth and GroundTruth.exists(folder) else None
    if args.truth and truth is None:
        print("no ground-truth files in", folder, "- drawing without truth")
    if args.tracks:
        image_paths = render_with_tracks(args, folder, calib, scans, list(picks), truth, out)
        if args.gif is not None and args.gif > 0 and image_paths:
            make_gif(image_paths, out / f"{folder.name}_tracks.gif", args.gif)
        return
    pipe = make_pipeline(calib, args)
    image_paths = []
    frames = CameraFrames(folder) if args.camera else None
    for k in picks:
        rec = scans[k]
        img = pipe.builder.ingest(lidar_message(rec))
        hdr, rgb = frames.closest(img.t, pipe.config.refinement.max_time_gap_s) if frames else (None, None)
        result = pipe.process(img, hdr, rgb)
        objs = truth.objects_at(img.t) if truth else None
        focus = None
        if args.focus is not None and objs:
            hit = [o for o in objs if o.id == args.focus]
            focus = (float(hit[0].centre[0]), float(hit[0].centre[1]), args.focus_radius) if hit else None
        image_path = out / f"{folder.name}_scan{k:05d}.png"
        write(image_path, result, calib, pipe.builder.geometry, rgb, objs, args.view_range, focus)
        image_paths.append(image_path)
    if args.gif is not None and args.gif > 0 and image_paths:
        make_gif(image_paths, out / f"{folder.name}.gif", args.gif)


if __name__ == "__main__":
    main()