"""Per-scan diagnostic of the size memory (step 6.11): why does a hazard keep an along error or a phantom speed?

    python -m testing.tools.tracking.extent_check --recording dev_default_seed12345
    python -m testing.tools.tracking.extent_check --recording dev_default_seed12345 --ids 4 5 --no-extent

--ids      hazard ids to follow (default: all);   --no-extent  memory off, for the comparison;   --scans A:B:S
--out      output folder, default testing/data/extent/<recording>

LiDAR only. For every scan in which a confirmed track matched a hazard it writes (extent_scans.csv) the true and the tracked centre,
the error along and across the line of sight, the speed, the extent of the visible cluster (hull in world axes, its depth along the
line of sight and the depth D the track uses), the remembered extent, and how many clusters were there. scene_clusters.csv has every
cluster of every scan (its hazard, hull, centre) and what the tracker did with it (matched, born, nothing) and the track's state. Also one plot per hazard
(extent_<id>.png): along error, speeds, remembered extent against the true size and the hull of the cluster, over time.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from server.tracking.extent import ExtentConfig, depth_along, hull
from server.tracking.track import TrackState
from testing.support.association_study import scan_measurements
from testing.support.fusion_study import replay_fusion
from testing.support.measurement_point_study import true_state
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import resolve


def collect(folder, data, extent: ExtentConfig, ids=None, scene: list = None) -> pd.DataFrame:
    rows = []

    def on_scan(t, tracker, res, items, truth):
        if scene is not None:                                          # every cluster of the scan and what the tracker did with it
            owner = {j: tid for tid, j, _ in res.matches}
            born = {j: tid for tid, j in res.births}
            state = {tr.id: (tr.state.name, tr.hits, tr.misses) for tr in tracker.tracks if hasattr(tr, "misses")} or {tr.id: (tr.state.name, tr.hits, -1) for tr in tracker.tracks}
            for j, (w, hid) in enumerate(items):
                lo, hi = hull(w)
                tid = owner.get(j, born.get(j))
                scene.append(dict(t=t, cluster=j, hazard=hid, how="match" if j in owner else "birth" if j in born else "none", track=tid,
                                  state=state.get(tid, ("", -1, -1))[0], track_hits=state.get(tid, ("", -1, -1))[1],
                                  lo_x=lo[0], lo_y=lo[1], hi_x=hi[0], hi_y=hi[1], cx=w.centre_xy[0], cy=w.centre_xy[1], range_m=w.range_m,
                                  merged=";".join(f"{a}>{b}" for a, b in res.merged), deleted=";".join(map(str, res.deleted))))
        by_id = {tr.id: tr for tr in tracker.tracks}
        for tid, j, _ in res.matches:
            tr, (w, hid) = by_id.get(tid), items[j]
            if tr is None or hid is None or tr.state != TrackState.CONFIRMED or (ids and hid not in ids):
                continue
            pos, vel = true_state(truth, hid, t)
            e = tr.position - pos
            u = np.array([np.cos(w.bearing), np.sin(w.bearing)])
            lo, hi = hull(w)
            ext = np.zeros(2) if tr.extent is None else np.asarray(tr.extent)
            rows.append(dict(id=hid, track=tid, t=t, true_x=pos[0], true_y=pos[1], x=tr.position[0], y=tr.position[1],
                             along=float(e @ u), cross=float(e @ np.array([-u[1], u[0]])), speed=float(np.hypot(*tr.velocity)),
                             true_speed=float(np.hypot(*vel)), hull_x=hi[0] - lo[0], hull_y=hi[1] - lo[1],
                             cluster_depth=float(w.depth_m), D=depth_along(ext, w.bearing, extent.prior_m) if tr.extent is not None else float("nan"),
                             ext_x=ext[0], ext_y=ext[1], bearing_deg=float(np.degrees(w.bearing)), n_clusters=len(items)))

    replay_fusion(folder, data, False, extent=extent, on_scan=on_scan)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, hid: int, out: Path):
    g = df[df["id"] == hid]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(g.t, g.along, label="along error"); ax[0].plot(g.t, g.cross, label="cross error"); ax[0].axhline(0, c="k", lw=.5); ax[0].legend(); ax[0].set_ylabel("m")
    ax[1].plot(g.t, g.speed, label="track speed"); ax[1].plot(g.t, g.true_speed, label="true speed"); ax[1].legend(); ax[1].set_ylabel("m/s")
    ax[2].plot(g.t, g.ext_x, label="memory x"); ax[2].plot(g.t, g.ext_y, label="memory y")
    ax[2].plot(g.t, g.hull_x, ":", label="cluster hull x"); ax[2].plot(g.t, g.hull_y, ":", label="cluster hull y")
    ax[2].plot(g.t, g.D, "--", c="k", label="D used"); ax[2].legend(ncol=3); ax[2].set_ylabel("m"); ax[2].set_xlabel("t (s)")
    fig.suptitle(f"hazard {hid}"); fig.tight_layout(); fig.savefig(out / f"extent_{hid}.png", dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--no-extent", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    folder = resolve(args.recording)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    data = scan_measurements(folder, PerceptionConfig(ground=GroundConfig(method=GroundMethod("range_image"))), scans, use_camera=False)
    scene = []
    df = collect(folder, data, ExtentConfig(enabled=not args.no_extent), args.ids, scene)
    out = Path(args.out) if args.out else DATA_DIR / "extent" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "extent_scans.csv", index=False)
    pd.DataFrame(scene).to_csv(out / "scene_clusters.csv", index=False)
    for hid in sorted(df["id"].unique()):
        plot(df, int(hid), out)
    print(df.groupby("id")[["along", "cross", "speed", "hull_x", "hull_y", "ext_x", "ext_y", "D", "cluster_depth"]].agg(["mean", "max"]).round(2).to_string())
    print("wrote", out)
    return df


if __name__ == "__main__":
    main()