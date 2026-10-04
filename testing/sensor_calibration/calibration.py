"""Calibration data sent by the simulator: ego geometry, sensor mounts, camera intrinsics, LiDAR beam tables.

Conventions (right-handed, same as the whole processor):
  vehicle frame: origin at the centre of the rear axle at ground level, x forward, y left, z up
  mount angles in degrees: yaw positive to the left, pitch positive up, roll by the right-hand rule about the
  forward axis, applied intrinsically in the order yaw, pitch, roll
  camera optical frame (for projection): x right, y down, z forward; pixel (0, 0) is the top-left corner of the
  image and pixel i covers [i, i + 1)
  range image: row 0 = highest beam, column 0 = leftmost (azimuth positive to the left)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_RETURN = 0              # dropout, fog, grazing: NOT free space
MAX_RANGE_OR_SKY = 65535   # the ray reached maximum range without a hit
RANGE_UNIT_M = 0.01


def rotation_from_euler(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation matrix vehicle_from_sensor for the given mount angles."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(-p), np.sin(-p)      # a positive (nose up) pitch is a negative rotation about the left axis
    cx, sx = np.cos(r), np.sin(r)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class Mount:
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Mount":
        return cls(d["x"], d["y"], d["z"], d.get("yawDeg", 0.0), d.get("pitchDeg", 0.0), d.get("rollDeg", 0.0))

    def rotation(self) -> np.ndarray:
        return rotation_from_euler(self.yaw_deg, self.pitch_deg, self.roll_deg)

    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_vehicle(self, points_sensor: np.ndarray) -> np.ndarray:
        """Points in the sensor body frame (x forward, y left, z up) to the vehicle frame."""
        return points_sensor @ self.rotation().T + self.translation()


@dataclass(frozen=True)
class EgoCalibration:
    wheelbase: float
    length: float
    width: float
    height: float
    rear_overhang: float
    front_overhang: float


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    rate_hz: float
    format: str
    fx: float
    fy: float
    cx: float
    cy: float
    horizontal_fov_deg: float
    vertical_fov_deg: float
    mount: Mount
    r_optical_from_vehicle: np.ndarray
    t_optical_from_vehicle: np.ndarray

    def project(self, p_vehicle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project vehicle-frame points (..., 3) to pixel coordinates. Returns u, v, depth."""
        p = np.asarray(p_vehicle, dtype=float) @ self.r_optical_from_vehicle.T + self.t_optical_from_vehicle
        depth = p[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * p[..., 0] / depth + self.cx
            v = self.fy * p[..., 1] / depth + self.cy
        return u, v, depth


@dataclass(frozen=True)
class LidarCalibration:
    rows: int
    cols: int
    rate_hz: float
    max_range_m: float
    mount: Mount
    elevations_deg: np.ndarray    # (rows,), row 0 = highest beam
    azimuths_deg: np.ndarray      # (cols,), column 0 = leftmost, positive to the left

    def directions(self) -> np.ndarray:
        """Unit ray directions in the sensor body frame, shape (rows, cols, 3)."""
        e = np.radians(self.elevations_deg)[:, None]
        a = np.radians(self.azimuths_deg)[None, :]
        return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e) + 0.0 * a], axis=-1)


@dataclass(frozen=True)
class Calibration:
    calibration_id: str
    ego: EgoCalibration
    camera: CameraCalibration
    lidar: LidarCalibration
    pose_rate_hz: float

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        e, c, l = d["ego"], d["camera"], d["lidar"]
        return cls(
            calibration_id=d["calibrationId"],
            ego=EgoCalibration(e["wheelbase"], e["length"], e["width"], e["height"], e["rearOverhang"], e["frontOverhang"]),
            camera=CameraCalibration(
                c["width"], c["height"], c["rateHz"], c["format"], c["fx"], c["fy"], c["cx"], c["cy"],
                c["horizontalFovDeg"], c["verticalFovDeg"], Mount.from_dict(c["mount"]),
                np.array(c["opticalFromVehicle"]["R"], dtype=float).reshape(3, 3),
                np.array(c["opticalFromVehicle"]["t"], dtype=float)),
            lidar=LidarCalibration(
                l["rows"], l["cols"], l["rateHz"], l["maxRangeM"], Mount.from_dict(l["mount"]),
                np.array(l["elevationsDeg"], dtype=float), np.array(l["azimuthsDeg"], dtype=float)),
            pose_rate_hz=d["poseRateHz"])

    @classmethod
    def from_json(cls, path_or_text) -> "Calibration":
        text = path_or_text
        if isinstance(path_or_text, Path) or (isinstance(path_or_text, str) and not path_or_text.lstrip().startswith("{")):
            text = Path(path_or_text).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))