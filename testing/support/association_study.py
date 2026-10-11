"""Tracker on a recording with ground truth (steps 6.4 and 6.5): are the measurements given to the right tracks?

`scan_measurements` runs the perception once and keeps, per scan, every cluster as a world cluster with the id of the
hazard it belongs to (the evaluation association of step 5.7; None for a cluster that belongs to no hazard).
`replay` then runs the Tracker (association with the gate, lifecycle, merge) over those scans and scores it against the truth.
A track takes the hazard of its first measurement; later the hazard of its last assigned measurement.

  correct      measurement assigned to a track whose last measurement came from the same hazard
  switches     measurement of another hazard assigned to the track (identity switch: the dangerous error)
  fragments    measurement that starts a new track although another measurement of the same hazard was assigned in that
               scan (a cluster split in two: the other part becomes a tentative track, as intended)
  duplicates   measurement that starts a new track although a live track of its hazard exists that got nothing in that scan
               (the gate rejected it: partial view, jump)
  born         tracks started;  confirmed  tracks that reached the confirmed state;  merged  tracks merged into another
  coverage     share of the (hazard, scan) pairs with a measurement in which a CONFIRMED track got that hazard's measurement
  d2_mean      mean squared Mahalanobis distance of the correct pairs (about 2 when the noise model fits) and d2_p95 its 95th
               percentile (chi-square 2 dof: 5.99)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from server.geometry.calibration import Calibration
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from server.tracking.association import AssociationConfig
from server.tracking.kalman import FilterConfig
from server.tracking.measurement import clusters_to_world
from server.tracking.noise import MeasurementNoise
from server.tracking.track import TrackConfig, TrackState
from server.tracking.tracker import Tracker, TrackerConfig
from testing.support.ground_truth import GroundTruth
from testing.support.perception_eval import EvalConfig, associate as truth_owner
from testing.support.replay_scans import run_pipeline


def scan_measurements(folder, config: PerceptionConfig = PerceptionConfig(), scans=None, use_camera: bool = False,
                      eval_cfg: EvalConfig = EvalConfig()) -> list:
    """[(t, [(WorldCluster, hazard id or None)])] per scan."""
    folder = Path(folder)
    pipe = PerceptionPipeline(Calibration.from_json(folder / "calibration.json"), config)
    truth = GroundTruth.load(folder)
    sensor_xy = pipe.builder.geometry.origin[:2]
    out = []
    for k, rec, result in run_pipeline(folder, pipe, use_camera=use_camera, scans=scans):
        img = result.image
        objs = truth.objects_at(img.t)
        owner = truth_owner(result.clusters, objs, eval_cfg)
        world = clusters_to_world(result.clusters, img, sensor_xy)
        out.append((img.t, [(w, objs[o].id if o is not None else None) for w, o in zip(world, owner)]))
    return out


def replay(scans: list, gate_chi2: float, noise: MeasurementNoise = MeasurementNoise(), point: str = "centre",
           sigma_a: float = 1.5, track: TrackConfig = TrackConfig()) -> dict:
    tracker = Tracker(TrackerConfig(point=point, association=AssociationConfig(gate_chi2=gate_chi2),
                                    filter=FilterConfig(sigma_a=sigma_a), track=track), noise)
    last_hid = {}                                     # track id -> hazard of its last measurement
    st = dict(measurements=0, correct=0, switches=0, fragments=0, duplicates=0, born=0, confirmed=0, merged=0)
    covered = total = 0
    d2_correct, events = [], []
    for t, items in scans:
        hids = [h for _, h in items]
        before = {tr.id for tr in tracker.tracks}
        res = tracker.step([w for w, _ in items], t)
        st["measurements"] += len(items)
        st["confirmed"] += len(res.confirmed)
        st["merged"] += len(res.merged)
        state = {tr.id: tr.state for tr in tracker.tracks}
        got = set()                                   # hazards that had a measurement assigned in this scan
        for tid, j, d2 in res.matches:
            h = hids[j]
            if h is None:
                continue
            prev = last_hid.get(tid)
            if prev is None or prev == h:
                st["correct"] += 1
                d2_correct.append(d2)
            else:
                st["switches"] += 1
                events.append((t, "switch", h, prev, d2))
            last_hid[tid] = h
            got.add(h)
        matched_ids = {tid for tid, _, _ in res.matches}
        for h in {h for h in hids if h is not None}:
            total += 1
            if any(hids[j] == h and state.get(tid) == TrackState.CONFIRMED for tid, j, _ in res.matches):
                covered += 1
        for tid, j in res.births:
            h = hids[j]
            last_hid[tid] = h
            st["born"] += 1
            if h is None:
                continue
            if h in got:
                st["fragments"] += 1
            elif any(last_hid.get(x) == h and x not in matched_ids for x in before if x in state):
                st["duplicates"] += 1
                events.append((t, "duplicate", h, h, float("nan")))
    d2 = np.array(d2_correct) if d2_correct else np.array([np.nan])
    st.update(coverage=covered / total if total else float("nan"), d2_mean=float(np.nanmean(d2)),
              d2_p95=float(np.nanpercentile(d2, 95)), gate_chi2=gate_chi2, events=events)
    return st