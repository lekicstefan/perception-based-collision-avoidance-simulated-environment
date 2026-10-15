"""Shows what the perception and the tracker make of a scene while the car approaches it (step 7.15 follow-up): per scan the clusters and the tracks.

  python -m testing.tools.risk.track_trace                       # the parked car beside the road, 10 m/s from x = 0
  python -m testing.tools.risk.track_trace --box 50 3.5 4.5 1.8 1.5 --speed 10 --scans 40

A single box that appears as several clusters, or tracks that are born and die while the box stands still, is what to look for.
"""
import argparse

from server.tracking.measurement import clusters_to_world
from server.tracking.outputs import track_output
from testing.support.closed_loop import CALIB, LIDAR
from testing.support.synthetic_scene import Scene


def main() -> None:
    from server.perception.pipeline import PerceptionPipeline
    from server.tracking.tracker import Tracker
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", nargs=5, type=float, default=[50.0, 3.5, 4.5, 1.8, 1.5], metavar=("X", "Y", "LEN", "WID", "HGT"))
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--scans", type=int, default=45)
    args = ap.parse_args()
    bx, by, bl, bw, bh = args.box
    pipe, tracker = PerceptionPipeline(CALIB), Tracker()
    seen_ids = set()
    print("   t      x  clusters (nearest point x,y / cells)                          tracks (id state hits x y)")
    for k in range(args.scans):
        t, x = k * 0.1, k * 0.1 * args.speed
        img, _, _ = Scene().add_box(bx - x, by, bl, bw, bh).render(LIDAR)
        img = type(img)(t, k, (x, 0.0, 0.0, 0.0, 0.0, 0.0), args.speed, 0.0, 0.0, img.range_m, img.state)
        result = pipe.process(img)
        world = clusters_to_world(result.clusters, img, pipe.builder.geometry.origin[:2])
        tracker.step(world, t)
        cl = "  ".join(f"({w.nearest_xy[0]:.1f},{w.nearest_xy[1]:.1f}/{w.n_cells})" for w in world) or "-"
        outs = [track_output(tr) for tr in tracker.tracks]
        seen_ids |= {o.id for o in outs}
        tr = "  ".join(f"#{o.id} {o.state[:3]} h{o.hits} ({o.position[0]:.1f},{o.position[1]:.1f})" for o in outs) or "-"
        print(f"{t:5.1f} {x:6.1f}  {cl:62.62}  {tr}")
    print(f"\ntrack ids seen: {sorted(seen_ids)}  (a standing single box should keep one id)")


if __name__ == "__main__":
    main()