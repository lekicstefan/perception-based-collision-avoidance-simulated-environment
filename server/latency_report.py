import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HOPS = [
    ("sensing        capture -> frame complete", "capture_ns", "ready_ns"),
    ("packing        complete -> built", "ready_ns", "built_ns"),
    ("hand-off       built -> sent (queue, network thread)", "built_ns", "sent_ns"),
    ("wire           sent -> received (Unity to Python)", "sent_ns", "recv_ns"),
    ("parsing        received -> parsed", "recv_ns", "parsed_ns"),
    ("processing     parsed -> answer ready", "parsed_ns", "done_ns"),
    ("reply          answer ready -> command sent", "done_ns", "cmd_sent_ns"),
    ("wire back      command sent -> received in Unity", "cmd_sent_ns", "cmd_received_ns"),
    ("apply          received -> applied in the car", "cmd_received_ns", "cmd_applied_ns"),
]


def read(path: Path) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def line(name, x):
    return f"{name:55s} {len(x):5d} {pct(x, 50):8.2f} {pct(x, 90):8.2f} {pct(x, 97):8.2f} {x.max() if len(x) else float('nan'):8.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--limit-ms", type=float, default=100.0, help="one frame period")
    ap.add_argument("--skip", type=int, default=5, help="ignore the first frames (start-up, Python imports and caches warming up)")
    args = ap.parse_args()
    run = Path(args.run)
    py = read(run / "python" / "latency.csv")
    un = read(run / "unity" / "latency.csv") if (run / "unity" / "latency.csv").exists() else {}
    if py:
        py = {k: v[args.skip:] for k, v in py.items()}
    if un:
        un = {k: v[args.skip:] for k, v in un.items()}
    if not py:
        sys.exit("no python/latency.csv rows")
    if not un:
        print("no unity/latency.csv (Unity writes it when the run ends): only the Python side is shown\n")

    n_unity = len(un["frame"]) if un else len(py["frame"])
    frames = py["frame"].astype(int)
    d = {k: v for k, v in py.items()}
    if un:
        idx = {int(f): i for i, f in enumerate(un["frame"])}
        sel = np.array([idx[f] for f in frames if f in idx])
        keep = np.array([f in idx for f in frames])
        d = {k: v[keep] for k, v in py.items()}
        for k in ("lidar_ready_ns", "camera_ready_ns", "cmd_received_ns", "cmd_applied_ns"):
            d[k] = un[k][sel]
        d["ready_ns"] = np.maximum(d["lidar_ready_ns"], d["camera_ready_ns"])
        # a missing event is 0 in the CSV
        d["cmd_received_ns"] = np.where(d["cmd_received_ns"] > 0, d["cmd_received_ns"], np.nan)
        d["cmd_applied_ns"] = np.where(d["cmd_applied_ns"] > 0, d["cmd_applied_ns"], np.nan)

    offset_ns = 0.0
    if un:
        # The two processes do not share a clock origin: Python's clock reads a constant offset from Unity's. The offset cancels in
        # the round trip, so it is estimated from it (wire time assumed equal both ways, error = half the difference, well under 1 ms).
        ok = ~np.isnan(d["cmd_received_ns"])
        w1 = d["recv_ns"][ok] - d["sent_ns"][ok]
        w2 = d["cmd_received_ns"][ok] - d["cmd_sent_ns"][ok]
        offset_ns = float(np.median((w1 - w2) / 2.0))
        for k in ("recv_ns", "parsed_ns", "done_ns", "cmd_sent_ns"):
            d[k] = d[k] - offset_ns
        print(f"clock offset Python minus Unity: {offset_ns / 1e9:.3f} s (removed below)\n")

    print(f"{'hop (ms)':55s} {'n':>5s} {'p50':>8s} {'p90':>8s} {'p97':>8s} {'max':>8s}")
    cols = {}
    if un:
        for name, k in (("  lidar scan done      capture -> lidar part ready", "lidar_ready_ns"),
                        ("  camera frame done    capture -> camera part ready", "camera_ready_ns")):
            x = (d[k] - d["capture_ns"]) / 1e6
            print(line(name, x[x > -1e6]))
    for name, a, b in HOPS:
        if a not in d or b not in d:
            continue
        x = (d[b] - d[a]) / 1e6
        x = x[~np.isnan(x)]
        cols[name] = x
        print(line(name, x))

    print()
    chain_names = [n for n, a, b in HOPS if n in cols]
    if un and all(n in cols for n in chain_names):
        applied = ~np.isnan(d["cmd_applied_ns"])
        total = (d["cmd_applied_ns"] - d["capture_ns"]) / 1e6
        proc = (d["done_ns"] - d["parsed_ns"]) / 1e6
        no_proc = total - proc
        lim = args.limit_ms
        lost = n_unity - len(frames)
        missing_cmd = int((~applied).sum())
        late_total = (np.nansum(total > lim) + lost + missing_cmd) / n_unity * 100
        late_np = (np.nansum(no_proc > lim) + lost + missing_cmd) / n_unity * 100
        print(line("age at applied: capture -> applied (as run)", total[applied]))
        print(line("age at applied without processing", no_proc[applied]))
        print(f"\nframes built by Unity {n_unity}, never received by Python {lost}, command never applied {missing_cmd} "
              f"(superseded by a newer one in the same step or lost)")
        print(f"late (age over {lim:g} ms, lost and never applied count as late): {late_total:.1f}% as run, {late_np:.1f}% without processing")
        base97, base90 = pct(no_proc[applied], 97), pct(no_proc[applied], 90)
        print(f"processing budget: {lim - base97:.1f} ms for 97% on time, {lim - pct(no_proc[applied], 90):.1f} ms for 90% on time "
              f"(limit {lim:g} ms minus the p97 / p90 age without processing)")

    arrivals = np.sort(d["recv_ns"]) / 1e6
    gaps = np.diff(arrivals)
    if len(gaps):
        print(f"\nbundle spacing at Python (ms): p50 {pct(gaps, 50):.1f}, p3 {pct(gaps, 3):.1f}, p97 {pct(gaps, 97):.1f}, max {gaps.max():.1f}")
    print(f"bundle size: {d['bytes'].mean() / 1e3:.0f} kB on average, {d['bytes'].max() / 1e3:.0f} kB at most")
    for name in cols:
        if name.startswith(("wire ", "wire back")) and len(cols[name]) and cols[name].min() < -0.5:
            print(f"WARNING: negative time in '{name.strip()}': the two clocks do not match, do not trust this table")


if __name__ == "__main__":
    main()