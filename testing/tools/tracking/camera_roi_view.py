"""Pictures of the camera bearing step (step 6.6): the search regions, expected extents and found edges per camera frame.

    python -m testing.tools.tracking.camera_roi_view --recording dev_default_seed12345 --t-from 5 --t-to 8 --gif 6
    python -m testing.tools.tracking.camera_roi_view --recording dev_default_seed12345 --t-from 9 --t-to 9.5 --camera-every 1

--recording     folder name inside testing/data/recordings, or a path (needs ground-truth files and camera frames)
--t-from/--t-to time window of the pictures in seconds (default: all)
--camera-every  use every n-th camera frame (default 3)
--ground        range_image (default) or height
--gif FPS       also make a gif of the pictures;   --out  output folder, default testing/data/camera_roi/<recording>

The tracker runs over the whole recording (LiDAR scans and camera frames in time order), pictures are written for the frames
in the window. What the pictures show: testing/support/camera_roi_view.py.
"""
import argparse
from pathlib import Path

from PIL import Image

from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from server.tracking.bearing import BearingConfig
from testing.support.association_study import scan_measurements
from testing.support.camera_roi_view import draw_rois
from testing.support.fusion_study import replay_fusion
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import resolve
from testing.support.scan_view import save_rgb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--t-from", type=float, default=float("-inf"))
    ap.add_argument("--t-to", type=float, default=float("inf"))
    ap.add_argument("--camera-every", type=int, default=3)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--gif", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    folder = resolve(args.recording)
    out = Path(args.out) if args.out else DATA_DIR / "camera_roi" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def on_camera(t, rgb, pose, diag, results, tracker):
        if args.t_from <= t <= args.t_to and diag:
            path = out / f"{folder.name}_t{t:07.3f}.png"
            save_rgb(path, draw_rois(rgb, diag, results, BearingConfig(), t))
            written.append(path)

    data = scan_measurements(folder, PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground))), None, use_camera=True)
    replay_fusion(folder, data, True, camera_every=args.camera_every, on_camera=on_camera)
    print(f"wrote {len(written)} pictures to {out}")
    if args.gif and written:
        frames = [Image.open(p).convert("RGB") for p in written]
        frames[0].save(out / f"{folder.name}.gif", save_all=True, append_images=frames[1:], duration=round(1000 / args.gif), loop=0)
        print("Created GIF:", out / f"{folder.name}.gif")
    return written


if __name__ == "__main__":
    main()