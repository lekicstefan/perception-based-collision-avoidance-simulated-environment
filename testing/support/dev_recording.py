"""Reader for the development recordings written by the simulator (DevRecorder, Phase 3).

Folder layout:
  meta.json              format version, scene, sensor configuration name, seed, rates, rows and columns
  sensors_resolved.json  the fully resolved sensor configuration used
  calibration.json       the calibration data (see avproc.calibration)
  lidar.bin              one packed little-endian record per scan (see lidar_dtype)
  camera/NNNNNN.jpg      one JPEG per camera frame, camera.csv is the index
  pose.csv               the standalone pose stream (what the processor is told)
  groundtruth files      ego.csv, hazards.csv, visibility.csv, hazard_info.csv (evaluation only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POSE_COLUMNS = ["t", "x", "y", "z", "yaw", "pitch", "roll", "speed", "yaw_rate", "steering"]


def lidar_dtype(rows: int, cols: int) -> np.dtype:
    """One LiDAR record: frame id, capture time, pose (x, y, z, yaw, pitch, roll),
    (speed, yaw rate, steering angle), range image in centimetres (0 = no return, 65535 = max range or sky)."""
    return np.dtype([
        ("frame", "<i4"),
        ("t", "<f8"),
        ("pose", "<f8", (6,)),
        ("motion", "<f4", (3,)),
        ("ranges", "<u2", (rows, cols)),
    ])


def load_meta(folder) -> dict:
    return json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))


def load_lidar(folder) -> np.ndarray:
    folder = Path(folder)
    meta = load_meta(folder)
    dtype = lidar_dtype(meta["rows"], meta["cols"])
    data = np.fromfile(folder / "lidar.bin", dtype=dtype)
    return data


def load_pose(folder) -> pd.DataFrame:
    return pd.read_csv(Path(folder) / "pose.csv")


def load_camera_index(folder) -> pd.DataFrame:
    return pd.read_csv(Path(folder) / "camera.csv")


def camera_path(folder, frame: int) -> Path:
    return Path(folder) / "camera" / f"{frame:06d}.jpg"