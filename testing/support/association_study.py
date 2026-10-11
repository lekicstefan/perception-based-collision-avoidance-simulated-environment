"""Association on a recording with ground truth (step 6.4): how many measurements go to the right track, for several gates.

`scan_measurements` runs the perception once and keeps, per scan, every cluster as a world measurement with the id of the
hazard it belongs to (the evaluation association of step 5.7; None for a cluster that belongs to no hazard).
`replay` then runs a bare tracker over those scans for one gate: a track is started from every measurement that is not
assigned, a track that gets no measurement for `max_coast_s` seconds is deleted (the real lifecycle is step 6.5), and the
choices of `associate` are scored against the truth:

  correct      measurement given to a track that was started by the same hazard
  switches     measurement given to a track of another hazard (identity switch: the dangerous error)
  duplicates   measurement not given to any track although a live track of its hazard exists (the gate was too tight or the
               track was lost): a second track of the same hazard starts
  tracks       tracks started in total; ideal is one per hazard
  d2_mean      mean squared Mahalanobis distance of the correct pairs (about 2 when the noise model is right, larger when
               R and the process noise are too small) and d2_p95 its 95th percentile (chi-square 2 dof: 5.99)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from server.geometry.calibration import Calibration
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from server.tracking.association import AssociationConfig, associate
from server.tracking.kalman import ConstantVelocityFilter, FilterConfig
from server.tracking.measurement import clusters_to_world
from server.tracking.noise import MeasurementNoise, measurement_of
from testing.support.ground_truth import GroundTruth
from testing.support.perception_eval import EvalConfig, associate as truth_owner
from testing.support.replay_scans import run_pipeline


@dataclass(frozen=True)
class ReplayConfig:
    point: str = "centre"
    sigma_a: float = 1.5
    max_coast_s: float = 1.5


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


def replay(scans: list, gate_chi2: float, noise: MeasurementNoise = MeasurementNoise(), cfg: ReplayConfig = ReplayConfig()) -> dict:
    fcfg = FilterConfig(sigma_a=cfg.sigma_a)
    tracks = []                                       # dicts: filter, hid (hazard that started it), last (time of the last update)
    stats = dict(measurements=0, correct=0, switches=0, duplicates=0, tracks=0)
    d2_correct, events = [], []
    acfg = AssociationConfig(gate_chi2=gate_chi2)
    for t, items in scans:
        tracks = [tr for tr in tracks if t - tr["last"] <= cfg.max_coast_s]
        meas = [measurement_of(w, noise, cfg.point) for w, _ in items]
        res = associate([tr["filter"] for tr in tracks], meas, acfg)
        stats["measurements"] += len(items)
        live = list(tracks)                                   # the tracks of this association (new ones are added below)
        for i, j in res.matches:
            tr, hid = tracks[i], items[j][1]
            if hid is not None and hid == tr["hid"]:
                stats["correct"] += 1
                d2_correct.append(res.d2[i, j])
            elif hid is not None:
                stats["switches"] += 1
                events.append((t, "switch", hid, tr["hid"], float(res.d2[i, j])))
            z, r, mt = meas[j]
            tr["filter"].update(z, r, mt)
            tr["last"] = t
        for j in res.unmatched_measurements:
            hid = items[j][1]
            if hid is not None and any(tr["hid"] == hid for tr in live):
                stats["duplicates"] += 1
                best = min(float(res.d2[i, j]) for i, tr in enumerate(live) if tr["hid"] == hid)
                events.append((t, "duplicate", hid, hid, best))
            z, r, mt = meas[j]
            tracks.append(dict(filter=ConstantVelocityFilter.initialise(z, r, mt, fcfg), hid=hid, last=t))
            stats["tracks"] += 1
    d2 = np.array(d2_correct) if d2_correct else np.array([np.nan])
    stats.update(d2_mean=float(np.nanmean(d2)), d2_p95=float(np.nanpercentile(d2, 95)), gate_chi2=gate_chi2)
    stats["events"] = events
    return stats