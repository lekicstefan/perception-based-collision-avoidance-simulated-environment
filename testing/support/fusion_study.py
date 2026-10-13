"""LiDAR and camera together on a recording with ground truth (step 6.6): does the camera bearing help the tracker?

`replay_fusion` runs the Tracker over the LiDAR scans of `scan_measurements` (step 6.4 module) and, when asked, over the camera
frames as well, in the order of their capture times (a scan at the same time first). The two inputs have their own rates and
time stamps: the camera frame is processed at its own capture time, with the pose of the car interpolated to that time.
After every scan the confirmed tracks that got a measurement of a hazard are compared with the truth:

  pos_rms     rms of the position error of the track against the true box centre (m); cross_rms and along_rms: the part
              across and along the line of sight from the sensor (the bearing can only help the first)
  vel_rms     rms of |track velocity - true velocity| (m/s)
  phantom_rms for a hazard that stands still: rms of the track speed (m/s)
  n           scans counted (the first `burn_in` of every hazard are not)

and about the camera: how many track-frames gave a measurement and why not (reasons of bearing.py), the number applied,
the mean normalised innovation squared of the applied updates (about 1 when sigma is right, chi-square 1 dof).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from server.geometry.calibration import Calibration
from server.tracking.association import AssociationConfig
from server.tracking.bearing import BearingConfig
from server.tracking.kalman import FilterConfig
from server.tracking.noise import MeasurementNoise
from server.tracking.extent import ExtentConfig
from server.tracking.outputs import OutputConfig
from server.tracking.track import TrackConfig, TrackState
from server.tracking.tracker import Tracker, TrackerConfig
from testing.support.dev_recording import camera_path
from testing.support.ground_truth import GroundTruth
from testing.support.measurement_point_study import true_state
from server.tracking.outputs import track_output
from testing.support.replay_scans import CameraFrames


def camera_timeline(folder, every: int = 1) -> list:
    """[(capture time, frame number, pose at that time)] of the recorded camera frames (every `every`-th)."""
    frames = CameraFrames(Path(folder))
    out = []
    for i in range(0, len(frames.index), max(1, every)):
        ct, fr = float(frames.index["t"].iloc[i]), int(frames.index["frame"].iloc[i])
        pose = tuple(float(np.interp(ct, frames.pose["t"], frames.pose[c])) for c in ("x", "y", "z", "yaw", "pitch", "roll"))
        out.append((ct, fr, pose))
    return out


def load_rgb(folder, frame: int):
    bgr = cv2.imread(str(camera_path(Path(folder), frame)))
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def replay_fusion(folder, scans: list, use_camera: bool, gate_chi2: float = 18.4, sigma_a: float = 1.4, point: str = "centre",
                  bearing: BearingConfig = BearingConfig(), camera_every: int = 1, burn_in: int = 10,
                  noise: MeasurementNoise = MeasurementNoise(), on_camera=None, output: OutputConfig = OutputConfig(),
                  track: TrackConfig = TrackConfig(), extent: ExtentConfig = ExtentConfig(), on_scan=None) -> dict:
    folder = Path(folder)
    truth = GroundTruth.load(folder)
    camera = Calibration.from_json(folder / "calibration.json").camera
    tracker = Tracker(TrackerConfig(point=point, association=AssociationConfig(gate_chi2=gate_chi2),
                                    filter=FilterConfig(sigma_a=sigma_a), bearing=bearing, output=output, track=track, extent=extent), noise)
    events = [(t, 0, i) for i, (t, _) in enumerate(scans)]
    cams = camera_timeline(folder, camera_every) if use_camera else []
    events += [(ct, 1, k) for k, (ct, _, _) in enumerate(cams)]
    events.sort()
    labels = {int(r["id"]): str(r["label"]) for _, r in truth.info.iterrows()}
    rows, reasons, nis, counted = [], Counter(), [], Counter()
    for t, kind, k in events:
        if kind == 1:
            _, frame, pose = cams[k]
            rgb = load_rgb(folder, frame)
            if rgb is None:
                continue
            diag = []
            results = tracker.step_camera(rgb, pose, t, camera, diag)
            if on_camera is not None:
                on_camera(t, rgb, pose, diag, results, tracker)
            for r in results:
                reasons[r.reason] += 1
                if r.nis is not None and r.reason in ("ok", "out_of_order", "same_time"):
                    nis.append(r.nis)
            continue
        _, items = scans[k]
        res = tracker.step([w for w, _ in items], t)
        by_id = {tr.id: tr for tr in tracker.tracks}
        if on_scan is not None:
            on_scan(t, tracker, res, items, truth)
        for tid, j, _ in res.matches:
            tr, (w, hid) = by_id.get(tid), items[j]
            if tr is None or hid is None or tr.state != TrackState.CONFIRMED:
                continue
            counted[hid] += 1
            if counted[hid] <= burn_in:
                continue
            pos, vel = true_state(truth, hid, t)
            out = track_output(tr, tracker.config.output)
            raw = tr.raw_velocity
            e = tr.position - pos
            u = np.array([np.cos(w.bearing), np.sin(w.bearing)])
            v = np.array([-u[1], u[0]])
            rows.append(dict(id=hid, label=labels.get(hid, str(hid)), t=t, pos_err=float(np.hypot(*e)), along=float(e @ u),
                             cross=float(e @ v), vel_err=float(np.hypot(*(tr.velocity - vel))),
                             speed=float(np.hypot(*tr.velocity)), true_speed=float(np.hypot(*vel)),
                             static=out.static, known=out.velocity_known, speed_out=float(np.hypot(*out.velocity_out)),
                             speed_sigma=out.speed_sigma, existence=out.existence,
                             raw_vel_err=float(np.hypot(*(raw - vel))) if raw is not None else float("nan"),
                             raw_speed=float(np.hypot(*raw)) if raw is not None else float("nan")))
    return dict(rows=pd.DataFrame(rows), reasons=dict(reasons), nis_mean=float(np.mean(nis)) if nis else float("nan"),
                camera_frames=len(cams))


COLUMNS = ["id", "label", "n", "true_speed", "pos_rms", "cross_rms", "along_rms", "vel_rms", "phantom_rms"]


def summarise_fusion(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = []
    for hid, g in rows.groupby("id"):
        static = float(g["true_speed"].mean()) < 0.1
        out.append(dict(id=int(hid), label=str(g["label"].iloc[0]), n=len(g), true_speed=float(g["true_speed"].mean()),
                        pos_rms=float(np.sqrt((g["pos_err"] ** 2).mean())), cross_rms=float(np.sqrt((g["cross"] ** 2).mean())),
                        along_rms=float(np.sqrt((g["along"] ** 2).mean())), vel_rms=float(np.sqrt((g["vel_err"] ** 2).mean())),
                        phantom_rms=float(np.sqrt((g["speed"] ** 2).mean())) if static else float("nan")))
    return pd.DataFrame(out)


OUTPUT_COLUMNS = ["id", "label", "n", "true_speed", "static_rate", "static_while_moving", "static_while_still", "leave_latency_s",
                  "known_rate", "speed_rms", "speed_out_rms", "speed_sigma", "existence"]
MOVING_MPS, STILL_MPS = 0.5, 0.1                  # a hazard faster than this moves, slower than this stands still


def summarise_outputs(rows: pd.DataFrame) -> pd.DataFrame:
    """Per hazard: how often the track is flagged static (should be about 1 for a hazard that stands still and 0 for a mover),
    how often its velocity is known, the rms speed of the filter and of the output after the static flag removed the phantom
    velocity, the mean speed standard deviation, the mean existence."""
    if rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = []
    for hid, g in rows.groupby("id"):
        g = g.sort_values("t")
        moving, still = g[g["true_speed"] > MOVING_MPS], g[g["true_speed"] <= STILL_MPS]
        out.append(dict(id=int(hid), label=str(g["label"].iloc[0]), n=len(g), true_speed=float(g["true_speed"].mean()),
                        static_rate=float(g["static"].mean()),
                        static_while_moving=float(moving["static"].mean()) if len(moving) else float("nan"),
                        static_while_still=float(still["static"].mean()) if len(still) else float("nan"),
                        leave_latency_s=leave_latency(g), known_rate=float(g["known"].mean()),
                        speed_rms=float(np.sqrt((g["speed"] ** 2).mean())), speed_out_rms=float(np.sqrt((g["speed_out"] ** 2).mean())),
                        speed_sigma=float(g["speed_sigma"].mean()), existence=float(g["existence"].mean())))
    return pd.DataFrame(out)


def leave_latency(g: pd.DataFrame) -> float:
    """Seconds from the moment a hazard that stood still starts to move until its track is no longer flagged static (0 when it
    never was static at that moment, NaN when the hazard does not go from still to moving in the rows)."""
    t, speed, static = g["t"].to_numpy(), g["true_speed"].to_numpy(), g["static"].to_numpy()
    for i in range(1, len(g)):
        if speed[i] > MOVING_MPS and speed[i - 1] <= STILL_MPS:
            for j in range(i, len(g)):
                if not static[j]:
                    return float(t[j] - t[i])
            return float(t[-1] - t[i])
    return float("nan")


def summarise_ablation(rows: pd.DataFrame) -> pd.DataFrame:
    """Filter against raw two-frame differencing (step 6.8): rms velocity error and, for hazards that stand still, rms speed (the
    phantom speed), and how many times larger the raw error is. A naive flag (raw speed under 0.5 m/s) is scored like the
    filter's: how often it calls a moving hazard static (dangerous) and a still hazard static."""
    cols = ["id", "label", "n", "true_speed", "filter_vel_rms", "raw_vel_rms", "raw_over_filter", "filter_phantom_rms",
            "raw_phantom_rms", "raw_static_while_moving", "raw_static_while_still"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for hid, g in rows.groupby("id"):
        g = g.dropna(subset=["raw_vel_err"])
        if g.empty:
            continue
        static = float(g["true_speed"].mean()) < STILL_MPS
        f, r = float(np.sqrt((g["vel_err"] ** 2).mean())), float(np.sqrt((g["raw_vel_err"] ** 2).mean()))
        flag = g["raw_speed"] < 0.5
        moving, still = g["true_speed"] > MOVING_MPS, g["true_speed"] <= STILL_MPS
        out.append(dict(id=int(hid), label=str(g["label"].iloc[0]), n=len(g), true_speed=float(g["true_speed"].mean()),
                        filter_vel_rms=f, raw_vel_rms=r, raw_over_filter=r / f if f > 0 else float("nan"),
                        filter_phantom_rms=float(np.sqrt((g["speed"] ** 2).mean())) if static else float("nan"),
                        raw_phantom_rms=float(np.sqrt((g["raw_speed"] ** 2).mean())) if static else float("nan"),
                        raw_static_while_moving=float(flag[moving].mean()) if moving.any() else float("nan"),
                        raw_static_while_still=float(flag[still].mean()) if still.any() else float("nan")))
    return pd.DataFrame(out, columns=cols)