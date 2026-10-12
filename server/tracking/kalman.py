"""Constant-velocity Kalman filter of one track (step 6.2).

State  x = [x, y, vx, vy]   in the world (start) frame, metres and metres per second.
Model  constant velocity with white-noise acceleration (discrete white-noise acceleration model of Bar-Shalom): in every
       interval the acceleration is a random constant of standard deviation sigma_a, independent between intervals.

           F(dt) = [[I, dt I], [0, I]]
           Q(dt) = sigma_a^2 [[dt^4/4 I, dt^3/2 I], [dt^3/2 I, dt^2 I]]            (I is the 2x2 identity)

       sigma_a is the one tuning number of the process noise. The same value is used for every object (no object is treated
       differently by its appearance); it is tuned against the ground truth in step 6.9. dt is the REAL time between
       the two time stamps, not the nominal period, so the filter works at any processing rate and after missed scans.
Bearing      the camera measures only a direction (step 6.6): theta = atan2(y - y_cam, x - x_cam) in the world frame, a scalar with
       standard deviation sigma. This is nonlinear in the state, so the update is an extended Kalman step: the measurement
       function is linearised at the predicted state, H_b = [-dy/r^2, dx/r^2, 0, 0] with dx, dy the offset of the track from the
       camera and r its length. The innovation is the wrapped angle difference. A bearing corrects the position across the line
       of sight and says nothing about the range; the time stamp rules below are the same as for a position.
Measurement  z = [x, y] (a position, step 6.3 chooses which point of the cluster), H = [[I, 0]], covariance R given by the
       caller per measurement (server/tracking/noise.py makes it from the sensor noise and the range).

Time stamps. A measurement is processed with the time it was captured:
  * later than the track        predict to the time stamp, then update (the normal case);
  * the same time (|dt| < 1 us) update without predicting (two sensors stamped with the same step; a repeated message
                                of the replay is a second look at the same instant and is simply a second update, exact
                                duplicates of a message are removed earlier by the sequence numbers);
  * earlier than the track      out of order: the track is stepped back to the time of the measurement (mean by F(-dt),
                                covariance by F P F^T + Q(|dt|)), updated there, and stepped forward again. This is the
                                one-step retrodiction approximation: it adds the process noise of the interval twice,
                                so it errs on the side of a larger covariance; it is used for a lateness of at most
                                `max_out_of_order_s`, older measurements are rejected and reported.
The covariance of the filter describes measurement and process noise only (section 5.5 of the specification); systematic
errors are covered by the inflation factor of the risk level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

H = np.hstack([np.eye(2), np.zeros((2, 2))])


@dataclass(frozen=True)
class FilterConfig:
    sigma_a: float = 1.4                    # m/s^2, standard deviation of the acceleration of an object (tuned in 6.9, was 1.5)
    initial_velocity_sigma: float = 10.0    # m/s, "large initial velocity uncertainty" of a new track
    max_out_of_order_s: float = 0.5         # an older measurement than this is rejected
    time_eps_s: float = 1e-6                # time stamps closer than this are the same instant


@dataclass(frozen=True)
class UpdateResult:
    applied: bool
    reason: str                             # "ok", "same_time", "out_of_order", "too_old"
    dt: float                               # measurement time minus track time before the call (s)
    innovation: np.ndarray | None = None    # (2,) z - H x_pred
    S: np.ndarray | None = None             # (2, 2) innovation covariance
    nis: float | None = None                # normalised innovation squared, y^T S^-1 y (chi-square, 2 degrees of freedom)


def transition(dt: float) -> np.ndarray:
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    return f


def process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """Q(dt) of the white-noise acceleration model for the state [x, y, vx, vy]."""
    dt = abs(dt)
    a2 = sigma_a ** 2
    q11, q12, q22 = dt ** 4 / 4.0, dt ** 3 / 2.0, dt ** 2
    q = np.zeros((4, 4))
    q[0, 0] = q[1, 1] = q11
    q[0, 2] = q[2, 0] = q[1, 3] = q[3, 1] = q12
    q[2, 2] = q[3, 3] = q22
    return a2 * q


def _propagate(x: np.ndarray, p: np.ndarray, dt: float, sigma_a: float):
    """State and covariance moved by dt (negative: stepped back). The process noise of |dt| is added in both directions."""
    f = transition(dt)
    return f @ x, f @ p @ f.T + process_noise(dt, sigma_a)


class ConstantVelocityFilter:
    def __init__(self, x: np.ndarray, p: np.ndarray, t: float, config: FilterConfig = FilterConfig()):
        self.x = np.asarray(x, dtype=float).copy()
        self.P = np.asarray(p, dtype=float).copy()
        self.t = float(t)
        self.config = config

    @classmethod
    def initialise(cls, z, r, t: float, config: FilterConfig = FilterConfig()) -> "ConstantVelocityFilter":
        """A new track from its first measurement: position z with covariance r, velocity zero with a large uncertainty."""
        x = np.array([z[0], z[1], 0.0, 0.0])
        p = np.zeros((4, 4))
        p[:2, :2] = r
        p[2, 2] = p[3, 3] = config.initial_velocity_sigma ** 2
        return cls(x, p, t, config)

    def copy(self) -> "ConstantVelocityFilter":
        return ConstantVelocityFilter(self.x, self.P, self.t, self.config)

    # ---- outputs --------------------------------------------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))

    @property
    def heading(self) -> float:
        """Direction of the velocity, rad from the world x axis, positive to the left (undefined when still: 0)."""
        return float(math.atan2(self.x[3], self.x[2]))

    # ---- prediction -----------------------------------------------------------------------------------------------------
    def predict(self, t: float) -> None:
        """Advance the track to time t (t >= the track time; an earlier t is a programming error)."""
        dt = float(t) - self.t
        if dt < -self.config.time_eps_s:
            raise ValueError(f"cannot predict backwards ({dt:.4f} s); use update() for an out-of-order measurement")
        if dt > self.config.time_eps_s:
            self.x, self.P = _propagate(self.x, self.P, dt, self.config.sigma_a)
            self.t = float(t)

    def predicted(self, t: float):
        """(state, covariance) at time t without changing the track; t may also be earlier than the track time."""
        dt = float(t) - self.t
        if abs(dt) <= self.config.time_eps_s:
            return self.x.copy(), self.P.copy()
        return _propagate(self.x, self.P, dt, self.config.sigma_a)

    def gate_distance2(self, z, r, t: float) -> float:
        """Squared Mahalanobis distance of a measurement from the prediction at time t (for the association gate)."""
        x, p = self.predicted(t)
        y = np.asarray(z, dtype=float) - H @ x
        s = H @ p @ H.T + np.asarray(r, dtype=float)
        return float(y @ np.linalg.solve(s, y))

    # ---- update ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _correct(x, p, z, r):
        y = np.asarray(z, dtype=float) - H @ x
        s = H @ p @ H.T + np.asarray(r, dtype=float)
        k = p @ H.T @ np.linalg.inv(s)
        i_kh = np.eye(4) - k @ H
        p_new = i_kh @ p @ i_kh.T + k @ np.asarray(r, dtype=float) @ k.T            # Joseph form: stays symmetric, positive
        p_new = 0.5 * (p_new + p_new.T)
        return x + k @ y, p_new, y, s

    def _apply_at(self, t: float, correct) -> UpdateResult:
        """Run `correct(x, p) -> (x, p, y, s)` at the time stamp t with the time rules of the module docstring."""
        cfg = self.config
        dt = float(t) - self.t
        if dt > cfg.time_eps_s:
            self.predict(t)
            reason = "ok"
        elif dt >= -cfg.time_eps_s:
            reason = "same_time"
        elif -dt <= cfg.max_out_of_order_s:
            x_b, p_b = _propagate(self.x, self.P, dt, cfg.sigma_a)                  # back to the time of the measurement
            x_b, p_b, y, s = correct(x_b, p_b)
            self.x, self.P = _propagate(x_b, p_b, -dt, cfg.sigma_a)                 # and forward again to the track time
            return UpdateResult(True, "out_of_order", dt, y, s, float(y @ np.linalg.solve(s, y)))
        else:
            return UpdateResult(False, "too_old", dt)
        self.x, self.P, y, s = correct(self.x, self.P)
        return UpdateResult(True, reason, dt, y, s, float(y @ np.linalg.solve(s, y)))

    def update(self, z, r, t: float) -> UpdateResult:
        """Process one position measurement z (2,) with covariance r (2, 2), captured at time t."""
        return self._apply_at(t, lambda x, p: self._correct(x, p, z, r))

    # ---- bearing (camera) -----------------------------------------------------------------------------------------------
    @staticmethod
    def _bearing_model(x, cam_xy):
        dx, dy = float(x[0] - cam_xy[0]), float(x[1] - cam_xy[1])
        r2 = max(dx * dx + dy * dy, 1e-6)
        return math.atan2(dy, dx), np.array([[-dy / r2, dx / r2, 0.0, 0.0]])

    @staticmethod
    def _correct_bearing(x, p, theta, sigma, cam_xy):
        h, hb = ConstantVelocityFilter._bearing_model(x, cam_xy)
        y = np.array([(float(theta) - h + math.pi) % (2.0 * math.pi) - math.pi])        # wrapped angle difference
        r = np.array([[float(sigma) ** 2]])
        s = hb @ p @ hb.T + r
        k = p @ hb.T @ np.linalg.inv(s)
        i_kh = np.eye(4) - k @ hb
        p_new = i_kh @ p @ i_kh.T + k @ r @ k.T
        return x + (k @ y), 0.5 * (p_new + p_new.T), y, s

    def gate_distance2_bearing(self, theta: float, sigma: float, cam_xy, t: float) -> float:
        """Squared normalised bearing innovation at time t (chi-square, 1 degree of freedom) without changing the track."""
        x, p = self.predicted(t)
        h, hb = self._bearing_model(x, cam_xy)
        y = (float(theta) - h + math.pi) % (2.0 * math.pi) - math.pi
        s = float((hb @ p @ hb.T)[0, 0]) + float(sigma) ** 2
        return y * y / s

    def update_bearing(self, theta: float, sigma: float, cam_xy, t: float) -> UpdateResult:
        """Extended Kalman update with the world bearing `theta` (rad, from the x axis, positive to the left) of the object as seen
        from the camera position `cam_xy` (world, metres), standard deviation `sigma` (rad), captured at time t."""
        return self._apply_at(t, lambda x, p: self._correct_bearing(x, p, theta, sigma, cam_xy))