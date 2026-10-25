import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    run = Path(args.run)
    path = run / "unity" / "commands.csv"
    if not path.exists():
        sys.exit(f"no {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    num = {k: np.array([float(r[k]) for r in rows]) for k in
           ("speed", "accel", "driver_accel", "applied_accel", "data_age_s", "object_id", "brake", "waiting", "remaining", "odometer")}
    rule = np.array([r["rule"] for r in rows])

    on = rule != "Driver"
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(int), [0]))))
    print(f"{len(t)} steps, {t[-1]:.1f} s, {num['odometer'][-1]:.0f} m\n")
    print("stretch        speed in -> out   brake set   peak decel   stopped at          waiting from..to")
    for a, b in zip(edges[::2], edges[1::2]):
        seg = slice(a, b)
        decel = -num["accel"][seg].min()
        stopped = np.flatnonzero(num["speed"][seg] < 0.05)
        at = ""
        if len(stopped):
            r = num["remaining"][a + stopped[0]]
            at = f"{r:+.2f} m to the stop"
        w = np.flatnonzero(num["waiting"][seg] > 0)
        wt = f"{t[a + w[0]]:.1f}..{t[a + w[-1]]:.1f} s" if len(w) else "-"
        print(f"{t[a]:6.1f}-{t[b - 1]:5.1f} s  {num['speed'][a]:5.2f} -> {num['speed'][b - 1]:5.2f}   "
              f"{num['brake'][seg].max():6.2f} m/s2  {decel:6.2f} m/s2   {at:18s}  {wt}")
    print(f"\npeak deceleration {-num['accel'].min():.2f} m/s2, peak acceleration {num['accel'].max():.2f} m/s2")
    print(f"age of the data when applied: median {np.median(num['data_age_s']) * 1000:.0f} ms, max {num['data_age_s'].max() * 1000:.0f} ms")
    print("'stopped at' is the distance to the stop position of the binding message when the speed first fell under 0.05 m/s (positive = short of it)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t, num["speed"])
    ax[0].set_ylabel("speed m/s")
    ax[1].plot(t, num["driver_accel"], label="driver", alpha=0.6)
    ax[1].plot(t, num["applied_accel"], label="requested by the system")
    ax[1].plot(t, num["accel"], label="car (after delay)")
    ax[1].set_ylabel("m/s2")
    ax[1].legend()
    ax[2].step(t, num["brake"], where="post", label="brake set m/s2")
    ax[2].step(t, num["waiting"], where="post", label="waiting (0/1)")
    ax[2].set_xlabel("time s")
    ax[2].legend()
    for a in ax:
        for e0, e1 in zip(edges[::2], edges[1::2]):
            a.axvspan(t[e0], t[e1 - 1], color="tab:red", alpha=0.08)
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = run / "drive.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()