"""Shows what the perception and the tracker make of a scene while the car approaches it (step 7.15 follow-up): per scan the clusters and the tracks.

  python -m testing.tools.risk.track_trace                       # the parked car beside the road, 10 m/s from x = 0
  python -m testing.tools.risk.track_trace --box 50 3.5 4.5 1.8 1.5 --speed 10 --scans 40
  python -m testing.tools.risk.track_trace --quiet               # only the why-blocks and the summary
  python -m testing.tools.risk.track_trace --quiet --extent-across-m 0.3   # with noise on the middle of the visible extent across the line of sight

A single box that appears as several clusters, or tracks that are born and die while the box stands still, is what to look for.
The tracker is built like the one of the processor (ProcessorConfig: gate, extent noise, process noise), so the trace shows what the closed loop sees.
Every scan after the first in which a track is BORN also prints a "why" block: for each existing track its predicted position and sigma, its
remembered extent and its misses, and for every cluster the point the track would measure it at and the squared Mahalanobis distance (the gate is
`gate_chi2`): a track that has a cluster of the box inside the gate would have taken it; one that does not is why the box got a second track.
"""
import argparse
import math

from server.risk.stage import StageConfig
from server.runtime.pipeline_processor import ProcessorConfig
from server.tracking.association import AssociationConfig
from server.tracking.extent import initial_extent, with_centre
from server.tracking.kalman import FilterConfig
from server.tracking.measurement import clusters_to_world
from server.tracking.noise import MeasurementNoise, measurement_of
from server.tracking.outputs import track_output
from server.tracking.tracker import TrackerConfig
from testing.support.closed_loop import CALIB, LIDAR
from testing.support.synthetic_scene import Scene


def why_lines(tracker, world, t: float) -> list:
    """For every track (before the step): prediction, extent memory, and for every cluster the measured point and d2 as the association sees it."""
    cfg, ecfg = tracker.config, tracker.config.extent
    lines = []
    for tr in tracker.tracks:
        x, p = tr.filter.predicted(t)
        ext = tr.eff_extent
        ext_txt = "none" if ext is None else f"({ext[0]:.1f},{ext[1]:.1f})"
        parts = []
        for j, w in enumerate(world):
            z, r, mt = measurement_of(with_centre(w, initial_extent(w, ecfg) if ext is None else ext, ecfg), tracker.noise, cfg.point)
            parts.append(f"c{j} z=({z[0]:.1f},{z[1]:.1f}) d2={tr.filter.gate_distance2(z, r, mt):.1f}")
        lines.append(f"        #{tr.id} {track_output(tr).state[:3]} pred=({x[0]:.1f},{x[1]:.1f}) sd=({math.sqrt(p[0, 0]):.2f},{math.sqrt(p[1, 1]):.2f}) "
                     f"ext={ext_txt} misses={tr.misses} | " + ("  ".join(parts) or "no clusters"))
    return lines


def main() -> None:
    from server.perception.pipeline import PerceptionPipeline
    from server.tracking.tracker import Tracker
    pc0 = ProcessorConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", nargs=5, type=float, default=[50.0, 3.5, 4.5, 1.8, 1.5], metavar=("X", "Y", "LEN", "WID", "HGT"))
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--scans", type=int, default=45)
    ap.add_argument("--pose-sigma-m", type=float, default=0.0)
    ap.add_argument("--extent-across-m", type=float, default=pc0.sigma_extent_across_m, help="MeasurementNoise.sigma_extent_across_m (m)")
    ap.add_argument("--quiet", action="store_true", help="no per-scan table, only the why-blocks of the births and the summary")
    args = ap.parse_args()
    bx, by, bl, bw, bh = args.box
    pc = ProcessorConfig()
    pipe = PerceptionPipeline(CALIB)
    tracker = Tracker(TrackerConfig(association=AssociationConfig(gate_chi2=pc.gate_chi2), filter=FilterConfig(sigma_a=StageConfig().sigma_a)),
                      MeasurementNoise(sigma_extent_m=pc.sigma_extent_m, sigma_extent_across_m=args.extent_across_m,
                                     pose_position_sigma_m=args.pose_sigma_m))
    seen_ids, births, lead_ids = set(), 0, []
    if not args.quiet:
        print("   t      x  clusters (nearest point x,y / cells)                          tracks (id state hits x y)")
    for k in range(args.scans):
        t, x = k * 0.1, k * 0.1 * args.speed
        img, _, _ = Scene().add_box(bx - x, by, bl, bw, bh).render(LIDAR)
        img = type(img)(t, k, (x, 0.0, 0.0, 0.0, 0.0, 0.0), args.speed, 0.0, 0.0, img.range_m, img.state)
        result = pipe.process(img)
        world = clusters_to_world(result.clusters, img, pipe.builder.geometry.origin[:2])
        why = why_lines(tracker, world, t)
        res = tracker.step(world, t)
        if world:                                           # which track holds the nearest cluster (the box itself, not its slivers)
            j0 = min(range(len(world)), key=lambda j: world[j].range_m)
            owner = next((m[0] for m in res.matches if m[1] == j0), None) or next((b[0] for b in res.births if b[1] == j0), None)
            if owner is not None and (not lead_ids or lead_ids[-1] != owner):
                lead_ids.append(owner)
        cl = "  ".join(f"({w.nearest_xy[0]:.1f},{w.nearest_xy[1]:.1f}/{w.n_cells})" for w in world) or "-"
        outs = [track_output(tr) for tr in tracker.tracks]
        seen_ids |= {o.id for o in outs}
        tr = "  ".join(f"#{o.id} {o.state[:3]} h{o.hits} ({o.position[0]:.1f},{o.position[1]:.1f})" for o in outs) or "-"
        if not args.quiet:
            print(f"{t:5.1f} {x:6.1f}  {cl:62.62}  {tr}")
        if k > 0 and res.births:
            births += len(res.births)
            print(f"    t={t:.1f} BORN {[b[0] for b in res.births]} from clusters {[b[1] for b in res.births]}; before this scan:")
            print("\n".join(why) if why else "        no tracks")
    print(f"\ntrack ids seen: {sorted(seen_ids)}  births after the first scan: {births}")
    print(f"the nearest cluster was held by: {' -> '.join('#' + str(i) for i in lead_ids)}   ({max(len(lead_ids) - 1, 0)} changes; a standing box should keep one id)")


if __name__ == "__main__":
    main()