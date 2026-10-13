"""Delay accounting (step 7.2): the command acts at the time of effect, not at the time of the data.

The state estimate refers to the data time t_d. By the time the command is sent and has reached the car, the car has driven on and every
object has moved. Before the risk is computed the ego car and every track are therefore extrapolated to

    t_e = t_d + smoothed processing latency + transit time

so that the delay t of the stopping model (stopping.py) holds only the actuation delay and the jitter margin: processing delay is never
counted twice.

  `DelayAccounting`   the smoothed latency (exponential average of what the cycles measured) and the time of effect. A sample is the time
                      from the time stamp of the newest data to the moment the command is sent (waiting for the cycle + processing); the
                      cycle runner supplies it (step 7.13).
  `predict_track`     a track output moved to t_e with the constant-velocity model and the same process noise as the filter (the
                      covariance grows with the interval), and expressed in the vehicle frame of the data time, where the corridor lives.
  `ego_travel`        how far the car drives along its path meanwhile, at its present speed (held constant: the car may be braking
                      already, but the command that would make it brake harder is not yet in effect, and a lower real speed only makes the
                      cap conservative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from server.geometry.frames import rotation
from server.tracking.kalman import process_noise, transition


@dataclass(frozen=True)
class DelayConfig:
    initial_latency_s: float = 0.05     # smoothed latency before the first sample
    smoothing: float = 0.2              # weight of a new sample in the exponential average
    transit_s: float = 0.005            # command transit time, loopback; PLACEHOLDER until measured (phase 10)
    max_latency_s: float = 0.5          # a sample above this is limited (a stalled cycle must not push the horizon far out)


class DelayAccounting:
    def __init__(self, cfg: DelayConfig = DelayConfig()):
        self.cfg = cfg
        self.latency_s = cfg.initial_latency_s

    def update(self, sample_s: float) -> None:
        """One measured latency (data time stamp to command sent)."""
        s = min(max(float(sample_s), 0.0), self.cfg.max_latency_s)
        self.latency_s += self.cfg.smoothing * (s - self.latency_s)

    @property
    def horizon_s(self) -> float:
        """Time from the data time to the expected time of effect."""
        return self.latency_s + self.cfg.transit_s

    def time_of_effect(self, t_data: float) -> float:
        return float(t_data) + self.horizon_s


@dataclass(frozen=True)
class PredictedTrack:
    """A track at the time of effect, in the vehicle frame of the data time."""
    id: int
    t: float                    # time of effect
    position: np.ndarray        # (2,)
    velocity: np.ndarray        # (2,)
    cov: np.ndarray             # (4, 4) state covariance [x, y, vx, vy]
    static: bool
    velocity_known: bool
    state: str
    existence: float
    age_s: float

    @property
    def pos_cov(self) -> np.ndarray:
        return self.cov[:2, :2]

    @property
    def vel_cov(self) -> np.ndarray:
        return self.cov[2:, 2:]


def predict_track(out, t_effect: float, pose_data, sigma_a: float) -> PredictedTrack:
    """`out` is a TrackOutput (world frame, at its own time out.t); `pose_data` the ego pose (x, y, z, yaw, ...) at the data time, which
    defines the vehicle frame the result is in. `sigma_a` is the process noise of the tracker's filter."""
    dt = max(float(t_effect) - float(out.t), 0.0)
    x = np.concatenate([np.asarray(out.position, dtype=float), np.asarray(out.velocity, dtype=float)])
    f = transition(dt)
    x, p = f @ x, f @ np.asarray(out.cov, dtype=float) @ f.T + process_noise(dt, sigma_a)
    r = rotation(float(pose_data[3]))                                    # vehicle -> world; the way back is its transpose
    b = np.zeros((4, 4))
    b[:2, :2] = b[2:, 2:] = r.T
    pos = r.T @ (x[:2] - np.array([pose_data[0], pose_data[1]]))
    return PredictedTrack(id=out.id, t=float(t_effect), position=pos, velocity=r.T @ x[2:], cov=b @ p @ b.T, static=bool(out.static),
                          velocity_known=bool(out.velocity_known), state=out.state, existence=float(out.existence), age_s=float(out.age_s))


def ego_travel(speed_mps: float, dt: float) -> float:
    """Distance (m) the car drives along its path in dt at constant speed."""
    return max(float(speed_mps), 0.0) * max(float(dt), 0.0)