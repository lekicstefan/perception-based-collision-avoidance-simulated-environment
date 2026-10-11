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
from server.tracking.track import TrackState
from server.tracking.tracker import Tracker, TrackerConfig
from testing.support.dev_recording import camera_path
from testing.support.ground_truth import GroundTruth
from testing.support.measurement_point_study import true_state
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


def replay_fusion(folder, scans: list, use_camera: bool, gate_chi2: float = 9.21, sigma_a: float = 1.5, point: str = "centre",
                  bearing: BearingConfig = BearingConfig(), camera_every: int = 1, burn_in: int = 10,
                  noise: MeasurementNoise = MeasurementNoise(), on_camera=None) -> dict:
    folder = Path(folder)
    truth = GroundTruth.load(folder)
    camera = Calibration.from_json(folder / "calibration.json").camera
    tracker = Tracker(TrackerConfig(point=point, association=AssociationConfig(gate_chi2=gate_chi2),
                                    filter=FilterConfig(sigma_a=sigma_a), bearing=bearing), noise)
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
        for tid, j, _ in res.matches:
            tr, (w, hid) = by_id.get(tid), items[j]
            if tr is None or hid is None or tr.state != TrackState.CONFIRMED:
                continue
            counted[hid] += 1
            if counted[hid] <= burn_in:
                continue
            pos, vel = true_state(truth, hid, t)
            e = tr.position - pos
            u = np.array([np.cos(w.bearing), np.sin(w.bearing)])
            v = np.array([-u[1], u[0]])
            rows.append(dict(id=hid, label=labels.get(hid, str(hid)), t=t, pos_err=float(np.hypot(*e)), along=float(e @ u),
                             cross=float(e @ v), vel_err=float(np.hypot(*(tr.velocity - vel))),
                             speed=float(np.hypot(*tr.velocity)), true_speed=float(np.hypot(*vel))))
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