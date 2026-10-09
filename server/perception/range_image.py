"""Range image ingestion (steps 5.1 and 5.2).

Turns a LIDAR message (uint16 centimetres) into a RangeImage the rest of the perception pipeline works on:

  * metres, as float32. Only cells that really hold a return have a number; every other cell is NaN, so a missing
    return can never be mistaken for a short or a long distance in later arithmetic.
  * a per-cell class (CellState), the validity semantics of the specification:
        VALID      a return: the ray hit something
        NO_RETURN  code 0: dropout, fog or grazing angle. This is NOT free space, it is unknown
        MAX_RANGE  code 65535: the ray reached maximum range or the sky. Free space up to the maximum range
    Treating NO_RETURN as free space is the unsafe interpretation, so the classes stay distinct until the
    occlusion logic (step 7.10) decides what each means.
  * the geometry tables of the calibrated sensor (LidarGeometry): ray directions in the vehicle frame, and the angle
    between neighbouring rays, which the segmentation (step 5.4) needs.

Row and column mapping (from the calibration): row 0 = highest beam, column 0 = leftmost (azimuth positive to the
left). Vehicle frame: origin at the centre of the rear axle at ground level, x forward, y left, z up.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from server.communication import protocol as P
from server.geometry.calibration import MAX_RANGE_OR_SKY, NO_RETURN, RANGE_UNIT_M, LidarCalibration


class CellState(IntEnum):
    NO_RETURN = 0
    VALID = 1
    MAX_RANGE = 2


@dataclass(frozen=True, eq=False)
class RangeImage:
    t: float                       # capture time (s)
    seq: int                       # sequence number of the LIDAR message
    pose: tuple                    # ego pose at capture: x, y, z, yaw, pitch, roll (start frame)
    speed: float                   # m/s
    yaw_rate: float                # rad/s
    steering_angle: float          # rad
    range_m: np.ndarray            # float32 (rows, cols), NaN unless state == VALID
    state: np.ndarray              # uint8 (rows, cols), values of CellState

    @property
    def shape(self) -> tuple:
        return self.range_m.shape

    @property
    def valid(self) -> np.ndarray:
        return self.state == CellState.VALID

    def counts(self) -> dict:
        """Number of cells per class, for logging."""
        return {s.name.lower(): int(np.count_nonzero(self.state == s)) for s in CellState}


class LidarGeometry:
    """Fixed tables of the calibrated sensor. Built once, reused for every scan."""

    def __init__(self, lidar: LidarCalibration):
        self.rows, self.cols = lidar.rows, lidar.cols
        self.max_range_m = lidar.max_range_m
        d_sensor = lidar.directions()                                   # (rows, cols, 3), sensor body frame
        self.origin = lidar.mount.translation()                         # sensor position in the vehicle frame
        self.dirs_vehicle = d_sensor @ lidar.mount.rotation().T         # (rows, cols, 3), unit vectors, vehicle frame
        # angle (rad) between the rays of neighbouring cells
        self.alpha_h = self._angle(d_sensor[:, :-1], d_sensor[:, 1:])   # (rows, cols - 1): cell (r, c) and (r, c + 1)
        self.alpha_v = self._angle(d_sensor[:-1], d_sensor[1:])         # (rows - 1, cols): cell (r, c) and (r + 1, c)

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.arccos(np.clip(np.einsum("...i,...i->...", a, b), -1.0, 1.0))

    def points_vehicle(self, img: RangeImage) -> np.ndarray:
        """3D point of every cell in the vehicle frame, shape (rows, cols, 3). NaN where the cell has no return."""
        return self.origin + img.range_m[..., None].astype(np.float64) * self.dirs_vehicle


class RangeImageBuilder:
    def __init__(self, lidar: LidarCalibration):
        self.lidar = lidar
        self.geometry = LidarGeometry(lidar)

    def convert(self, h: P.Header, raw_ranges: np.ndarray) -> RangeImage:
        """h: the parsed header of the message, raw_ranges: its uint16 range image (rows, cols)."""
        expected = (self.lidar.rows, self.lidar.cols)
        if raw_ranges.shape != expected:
            raise P.ProtocolError(f"LIDAR image is {raw_ranges.shape}, the calibration says {expected}")
        state = np.full(expected, CellState.VALID, dtype=np.uint8)
        state[raw_ranges == NO_RETURN] = CellState.NO_RETURN
        state[raw_ranges == MAX_RANGE_OR_SKY] = CellState.MAX_RANGE
        range_m = raw_ranges.astype(np.float32) * np.float32(RANGE_UNIT_M)
        range_m[state != CellState.VALID] = np.nan
        return RangeImage(h.t, h.seq, h.pose, h.speed, h.yaw_rate, h.steering_angle, range_m, state)

    def ingest(self, raw_message: bytes) -> RangeImage:
        """From the raw bytes of a LIDAR message (as delivered by the transport)."""
        h, ranges = P.parse_lidar(raw_message)
        return self.convert(h, ranges)