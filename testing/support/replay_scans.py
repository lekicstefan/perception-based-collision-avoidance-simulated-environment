"""Feeding the scans of a development recording through the perception pipeline (for tools and evaluation)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from server.communication import protocol as P
from testing.support.dev_recording import camera_path, load_camera_index, load_lidar, load_pose
from testing.support.paths import RECORDINGS_DIR


def resolve(recording) -> Path:
    """A folder name inside testing/data/recordings, or a path."""
    p = Path(recording)
    return p if p.exists() else RECORDINGS_DIR / recording


def lidar_message(rec) -> bytes:
    """The LIDAR message of one record of lidar.bin."""
    return P.build_message(P.MsgType.LIDAR, int(rec["frame"]), P.lidar_payload(rec["ranges"]), t=float(rec["t"]),
                           pose=tuple(rec["pose"]), speed=float(rec["motion"][0]), yaw_rate=float(rec["motion"][1]),
                           steering_angle=float(rec["motion"][2]))


class CameraFrames:
    """The recorded camera frames of a folder: the one closest in time, with the pose from pose.csv at that time."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.index = load_camera_index(folder)
        self.pose = load_pose(folder)

    def closest(self, t: float, max_gap_s: float):
        """(header-like with pose and t, rgb image) or (None, None)."""
        i = int(np.argmin(np.abs(self.index["t"].to_numpy() - t)))
        ct = float(self.index["t"].iloc[i])
        if abs(ct - t) > max_gap_s:
            return None, None
        bgr = cv2.imread(str(camera_path(self.folder, int(self.index["frame"].iloc[i]))))
        if bgr is None:
            return None, None
        pose = tuple(float(np.interp(ct, self.pose["t"], self.pose[c])) for c in ("x", "y", "z", "yaw", "pitch", "roll"))
        return SimpleNamespace(t=ct, pose=pose), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_pipeline(folder: Path, pipeline, use_camera: bool = False, scans=None):
    """Yields (scan index, record, PerceptionResult) for the scans of a recording (all of them, or the given indices)."""
    records = load_lidar(folder)
    frames = CameraFrames(folder) if use_camera else None
    for k in (range(len(records)) if scans is None else scans):
        rec = records[k]
        img = pipeline.builder.ingest(lidar_message(rec))
        hdr, rgb = frames.closest(img.t, pipeline.config.refinement.max_time_gap_s) if frames else (None, None)
        yield k, rec, pipeline.process(img, hdr, rgb)