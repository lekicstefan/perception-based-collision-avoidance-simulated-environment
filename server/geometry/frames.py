"""Frames of reference and conversions (step 6.1). The one place where the coordinate convention lives.

The processor uses ONE convention everywhere (right-handed):

    x forward, y left, z up;   yaw positive to the left (counter-clockwise seen from above), pitch positive nose up,
    roll by the right-hand rule about the forward axis (positive: the right side goes down).

Three frames:

    sensor frame   -> vehicle frame   fixed calibration (server/geometry/calibration.py)
    vehicle frame  -> world frame     pose of the car at the capture time of the measurement
    world frame    = start frame      the vehicle frame at the beginning of the run (origin and heading of the car at t = 0)

The pose that arrives in every message header is already a start-frame pose in this convention: the simulator converts
at its boundary, in `PoseEstimator.ToStartFrame` and `YawToStart` (Unity script). Unity is left-handed, Y up, x right,
z forward; its yaw about Y is clockwise seen from above. `unity_to_processor` and `to_start_frame` below are an independent
Python statement of the same conversion; the test vectors of testing/tests/server/geometry/test_frames.py check them
against a line-by-line port of the C# formulas. They are not called during a run.

Tracking is 2D on the ground plane. Heights are used for detection and size, not for risk, so the world transform here is
a rotation by the yaw and a shift by (x, y); pitch and roll of the car are ignored (they change a ground position by a
fraction of a percent at the angles of a car on a road).

All functions accept a single point (2,) or an array (..., 2); angles are radians.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class Pose(NamedTuple):
    """Car pose in the start frame. A plain tuple of this order is what the message headers carry (`Header.pose`)."""
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float


def wrap_angle(a):
    """Angle wrapped to (-pi, pi]; works on numbers and arrays."""
    w = (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    w = np.where(w <= -math.pi, w + 2.0 * math.pi, w)
    return float(w) if np.ndim(w) == 0 else w


def rotation(yaw: float) -> np.ndarray:
    """2x2 matrix that turns a vector by `yaw` to the left (counter-clockwise)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def vehicle_to_world(xy, pose) -> np.ndarray:
    """Positions in the vehicle frame of the capture time -> world (start frame). pose = (x, y, z, yaw, pitch, roll)."""
    p = np.asarray(xy, dtype=float)
    r = rotation(pose[3])
    return p @ r.T + np.array([pose[0], pose[1]])


def world_to_vehicle(xy, pose) -> np.ndarray:
    """Positions in the world frame -> vehicle frame of the car at `pose` (the inverse of vehicle_to_world)."""
    p = np.asarray(xy, dtype=float) - np.array([pose[0], pose[1]])
    return p @ rotation(pose[3])


def rotate_to_world(v, pose) -> np.ndarray:
    """Vectors (velocity, direction, offset: no shift) from the vehicle frame to the world frame."""
    return np.asarray(v, dtype=float) @ rotation(pose[3]).T


def rotate_to_vehicle(v, pose) -> np.ndarray:
    return np.asarray(v, dtype=float) @ rotation(pose[3])


def rotate_covariance(cov, yaw: float) -> np.ndarray:
    """Covariance of a 2D quantity turned by `yaw`: R C R^T. A measurement covariance given in the vehicle frame (for
    example along and across the line of sight) is moved to the world frame with the yaw of the capture pose."""
    r = rotation(yaw)
    return r @ np.asarray(cov, dtype=float) @ r.T


def line_of_sight_covariance(bearing: float, sigma_along: float, sigma_across: float) -> np.ndarray:
    """2x2 covariance with `sigma_along` along the direction `bearing` and `sigma_across` perpendicular to it (the
    shape of a LiDAR range and bearing error). `bearing` is an angle in the frame the result is wanted in."""
    return rotate_covariance(np.diag([sigma_along ** 2, sigma_across ** 2]), bearing)


def bearing_of(xy, origin=(0.0, 0.0)) -> float:
    """Direction from `origin` to the point, rad, positive to the left (counter-clockwise from the x axis)."""
    d = np.asarray(xy, dtype=float) - np.asarray(origin, dtype=float)
    return math.atan2(d[1], d[0])


# ---- the conversion at the simulator boundary (reference and test vectors) ------------------------------------------
def unity_to_processor(position_unity, euler_deg_unity) -> Pose:
    """A Unity pose to the processor convention, before it is made relative to the start pose.

    Unity (left-handed): x right, y up, z forward; `euler_deg_unity` = (pitch about x, yaw about y, roll about z) as in
    Transform.eulerAngles, degrees (0 to 360 is fine). Positive Unity yaw turns the nose to the RIGHT (clockwise seen from above), positive
    Unity pitch tips the nose DOWN, positive Unity roll tips the car to the LEFT. All three signs flip:

        x_p = z_u,  y_p = -x_u,  z_p = y_u
        yaw_p = -yaw_u,  pitch_p = -pitch_u,  roll_p = -roll_u
    """
    xu, yu, zu = (float(v) for v in position_unity)
    pu, yw, ru = (float(v) for v in euler_deg_unity)
    return Pose(zu, -xu, yu, wrap_angle(-math.radians(yw)), wrap_angle(-math.radians(pu)), wrap_angle(-math.radians(ru)))


def unity_vector_to_processor(v_unity) -> np.ndarray:
    """A Unity horizontal vector (x right, z forward), for example a velocity, to the processor (x forward, y left)."""
    return np.array([float(v_unity[2]), -float(v_unity[0])])


def to_start_frame(pose: Pose, start: Pose) -> Pose:
    """A pose in the processor convention expressed in the start frame (the frame of the car pose `start`).
    The start pose itself becomes (0, 0, 0, 0, pitch, roll); height, pitch and roll are kept as they are, only the
    ground position and the heading are made relative."""
    xy = world_to_vehicle((pose[0], pose[1]), start)
    return Pose(float(xy[0]), float(xy[1]), pose[2], wrap_angle(pose[3] - start[3]), pose[4], pose[5])