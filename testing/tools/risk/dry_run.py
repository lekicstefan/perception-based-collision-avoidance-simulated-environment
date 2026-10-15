"""Dry run of the whole chain without Unity (step 7.15): scripted scenes through the real perception, tracker and risk stage into a simple car.

  python -m testing.tools.risk.dry_run                 # the table of the scenes
  python -m testing.tools.risk.dry_run --csv out.csv   # plus one row per scan of every scene
  python -m testing.tools.risk.dry_run --trace parked  # every scan of the scene whose name contains 'parked'

Prints for each scene: collided, stopped, the final gap and speed, the strongest braking request, whether an emergency happened, and the
p50 / p95 / max wall time per scan (perception, tracking and stage together) against the 33 ms cycle.
"""
import argparse
import csv

from testing.support.closed_loop import Box, LoopConfig, run_loop

SCENES = {
    "empty road": ([], LoopConfig(start_speed_mps=10.0, duration_s=6.0)),
    "stopped car 80 m": ([Box(80.0, 0.0)], LoopConfig()),
    "stopped car 40 m": ([Box(40.0, 0.0)], LoopConfig()),
    "pedestrian-size 35 m": ([Box(35.0, 0.0, 0.5, 0.5, 1.7)], LoopConfig(start_speed_mps=8.0, desired_mps=8.0)),
    "parked car beside": ([Box(50.0, 3.5)], LoopConfig(start_speed_mps=10.0, desired_mps=10.0, duration_s=40.0)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--trace", default=None, help="print every scan of the scene whose name contains this text")
    args = ap.parse_args()
    rows = []
    print(f"{'scene':22} {'coll':>5} {'stop':>5} {'gap m':>7} {'v end':>6} {'maxbrk':>7} {'emerg':>6} {'p50 ms':>7} {'p95 ms':>7} {'max ms':>7}")
    for name, (boxes, cfg) in SCENES.items():
        r = run_loop(boxes, cfg)
        f = r.final
        print(f"{name:22} {str(r.collided):>5} {str(r.stopped):>5} {f['gap_m']:7.2f} {f['v']:6.2f} {r.max_brake:7.2f} {str(r.any_emergency):>6} "
              f"{r.percentile_ms('total_ms', 50):7.1f} {r.percentile_ms('total_ms', 95):7.1f} {r.percentile_ms('total_ms', 100):7.1f}")
        e = r.first_emergency()
        if e:
            print(f"    first emergency: t={e['t']:.1f} x={e['x']:.1f} v={e['v']:.2f} gap={e['gap_m']:.2f} cause={e['brake_cause']} "
                  f"d_path={e['brake_d_path_m']} d_eff={e['brake_d_eff_m']} tau={e['brake_tau_s']} caps={e['comfort_cap']:.1f}/{e['safety_cap']:.1f} "
                  f"risk={e['risk_level']}")
        if args.trace and args.trace in name:
            print("    t      x      v   comf   safe  brake em reg risk trk  cause            d_path   d_eff   tau")
            for q in r.rows:
                print(f"    {q['t']:5.1f} {q['x']:6.1f} {q['v']:6.2f} {q['comfort_cap']:6.1f} {q['safety_cap']:6.1f} {q['brake_request']:5.1f} "
                      f"{int(q['emergency']):2d} {q['regime']:3d} {q['risk_level']:4d} {q['n_tracks']:3d}  {str(q['brake_cause'])[:15]:15} "
                      f"{q['brake_d_path_m']} {q['brake_d_eff_m']} {q['brake_tau_s']}")
        rows += [{"scene": name, **row} for row in r.rows]
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()