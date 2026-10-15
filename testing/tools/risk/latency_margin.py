"""The jitter margin of the stopping model from a recorded run (step 7.15 follow-up; spec 5.6: "measured gap between the 99th and the 50th percentile").

  python -m testing.tools.risk.latency_margin runs/<id>

Reads python/commands.csv (processing_time of every command, written by the cycle runner) and python/cycles.csv (process_ms) and prints the
50th / 99th percentile and p99 - p50, which is what `StoppingParams.jitter_margin_s` should be (currently the placeholder 0.05 s). This is the
processor side only: the transit time from Python to Unity (`DelayConfig.transit_s`, placeholder 5 ms) and the Unity side of the delay have to
be measured in Unity (phase 10); add their jitter to this one.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def read(path: Path, column: str) -> np.ndarray:
    with open(path, newline="") as fh:
        return np.array([float(r[column]) for r in csv.DictReader(fh) if r.get(column) not in (None, "")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a run folder or its python folder")
    args = ap.parse_args()
    base = Path(args.run)
    base = base / "python" if (base / "python").is_dir() else base
    for name, col, scale, unit in (("commands.csv", "processing_time", 1e3, "ms"), ("cycles.csv", "process_ms", 1.0, "ms")):
        f = base / name
        if not f.exists():
            print(f"{name}: not found in {base}")
            continue
        v = read(f, col) * scale
        if len(v) < 20:
            print(f"{name}: only {len(v)} rows, too few")
            continue
        p50, p99 = np.percentile(v, 50), np.percentile(v, 99)
        print(f"{name:13} {col:16} n={len(v):5d}  p50 {p50:6.2f} {unit}  p99 {p99:6.2f} {unit}  max {v.max():6.2f} {unit}   p99 - p50 = {p99 - p50:6.2f} {unit}")
        if name == "commands.csv":
            print(f"  -> jitter_margin_s (processor side) = {(p99 - p50) / 1e3:.3f}   (placeholder in use: 0.050)")


if __name__ == "__main__":
    main()