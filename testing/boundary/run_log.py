"""Run folder logging on the Python side (docs/run_folder.md): runs/<run id>/python/.

The run folder is created by Unity and announced in the HELLO message (run_dir). Without Unity (replay, offline
tests) offline_dir() makes a folder of its own. CSV is used for the tables because it is written row by row and
survives a crash; it is converted to Parquet when the data is analysed (phase 16).
"""
from __future__ import annotations

import csv
import datetime
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("numpy", "scipy", "opencv-python", "opencv-python-headless", "opencv-contrib-python", "pyzmq", "numba",
            "pandas", "pyarrow", "matplotlib", "pytest")


def git_info(repo: Path = REPO_ROOT) -> dict:
    def git(*args):
        try:
            r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


class TableLog:
    """Append-only CSV, flushed about once a second."""

    def __init__(self, path: Path, columns: list[str], flush_every: float = 1.0):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(columns)
        self.flush_every = flush_every
        self._last_flush = time.monotonic()
        self.rows = 0

    def row(self, *values) -> None:
        self.w.writerow(values)
        self.rows += 1
        if time.monotonic() - self._last_flush > self.flush_every:
            self.f.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        if not self.f.closed:
            self.f.close()


class RunLog:
    def __init__(self, run_dir):
        self.dir = Path(run_dir) / "python"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tables: list[TableLog] = []

    @staticmethod
    def offline_dir(label: str = "python_only") -> Path:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return REPO_ROOT / "runs" / f"{stamp}_{label}"

    def table(self, name: str, columns: list[str]) -> TableLog:
        t = TableLog(self.dir / f"{name}.csv", columns)
        self.tables.append(t)
        return t

    def write_meta(self, **extra) -> None:
        meta = {
            "startedUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": package_versions(),
            "git": git_info(),
            "argv": sys.argv,
        }
        meta.update(extra)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def write_summary(self, **data) -> None:
        (self.dir / "summary.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def close(self) -> None:
        for t in self.tables:
            t.close()