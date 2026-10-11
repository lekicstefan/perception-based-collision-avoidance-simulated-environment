"""Which point of a cluster should the tracker measure? (step 6.3)

The cluster geometry offers three points: the nearest point (a real LiDAR return, but a point of the surface that moves
along the object when the viewing angle changes), the corrected centre (nearest point pushed back by half a size prior,
a model of the unseen depth) and the plain centroid of the returns (it moves when more or other cells are hit).
Every one of them moves a little from scan to scan even when the object stands still; a Kalman filter reads that as
velocity. This module measures it on a recording with ground truth:

  collect()   per scan and per hazard the largest cluster of the hazard, in the world frame (same association as the
              perception evaluation), with the true position and velocity of the hazard
  study()     per hazard and point: the same constant velocity filter (R from MeasurementNoise) fed with that point only,
              and the raw two-scan difference of the point, both against the true velocity

Numbers per hazard and point (after `burn_in` measurements, so that the start-up transient does not count):
  phantom_*   for a hazard that stands still (true speed < static_speed): mean, rms and maximum of the filter speed (m/s)
  vel_rms     rms of |filter velocity - true velocity| (m/s)       speed_rms  rms of filter speed - true speed
  raw_rms     the same for the difference of two consecutive measurements (what no filter would give)
  bias        length of the mean position error against the true box centre (m): a fixed offset, harmless for velocity
  offset_sd   standard deviation of the position error, x and y combined (m): this is what turns into phantom velocity
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from server.geometry.calibration import Calibration
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from server.tracking.kalman import ConstantVelocityFilter, FilterConfig
from server.tracking.measurement import WorldCluster, clusters_to_world
from server.tracking.noise import MeasurementNoise, POINTS, measurement_of
from testing.support.ground_truth import GroundTruth
from testing.support.perception_eval import EvalConfig, associate
from testing.support.replay_scans import run_pipeline


@dataclass(frozen=True)
class StudyConfig:
    sigma_a: float = 1.5
    burn_in: int = 10                       # measurements of a hazard that are not counted
    static_speed_mps: float = 0.1           # a hazard slower than this does not move
    max_gap_s: float = 1.0                  # a longer gap restarts the filter
    min_samples: int = 15                   # hazards with fewer measurements are left out of the summary


def true_state(truth: GroundTruth, hid: int, t: float):
    g = truth.hazards[truth.hazards["id"] == hid].sort_values("t")
    return (np.array([np.interp(t, g["t"], g["x"]), np.interp(t, g["t"], g["y"])]),
            np.array([np.interp(t, g["t"], g["vx"]), np.interp(t, g["t"], g["vy"])]))


def collect(folder, config: PerceptionConfig = PerceptionConfig(), scans=None, use_camera: bool = False,
            eval_cfg: EvalConfig = EvalConfig()) -> pd.DataFrame:
    """One row per hazard and scan in which the hazard was found: the world clusters (object column `w`) and the truth."""
    folder = Path(folder)
    pipe = PerceptionPipeline(Calibration.from_json(folder / "calibration.json"), config)
    truth = GroundTruth.load(folder)
    sensor_xy = pipe.builder.geometry.origin[:2]
    rows = []
    for k, rec, result in run_pipeline(folder, pipe, use_camera=use_camera, scans=scans):
        img = result.image
        objs = truth.objects_at(img.t)
        owner = associate(result.clusters, objs, eval_cfg)
        world = clusters_to_world(result.clusters, img, sensor_xy)
        best = {}
        for w, o in zip(world, owner):
            if o is not None and (o not in best or w.n_cells > best[o].n_cells):
                best[o] = w
        for o, w in best.items():
            pos, vel = true_state(truth, objs[o].id, img.t)
            rows.append(dict(scan=k, t=img.t, id=objs[o].id, label=objs[o].label, w=w, true_x=pos[0], true_y=pos[1],
                             true_vx=vel[0], true_vy=vel[1]))
    return pd.DataFrame(rows)


def run_filter(series: pd.DataFrame, point: str, noise: MeasurementNoise, cfg: StudyConfig) -> pd.DataFrame:
    """The filter and the raw difference on one hazard with one measurement point (series sorted by time)."""
    fcfg = FilterConfig(sigma_a=cfg.sigma_a)
    flt, out, prev, n, prev_z = None, [], None, 0, None
    for r in series.itertuples():
        z, cov, t = measurement_of(r.w, noise, point)
        if flt is None or t - flt.t > cfg.max_gap_s:
            flt, prev, n, prev_z = ConstantVelocityFilter.initialise(z, cov, t, fcfg), None, 0, None
        else:
            flt.update(z, cov, t)
        n += 1
        raw = (z - prev[0]) / (t - prev[1]) if prev is not None and t > prev[1] else np.full(2, np.nan)
        prev = (z, t)
        v = flt.velocity
        jump = float(np.hypot(*(z - prev_z))) if prev_z is not None else np.nan
        prev_z = z
        out.append(dict(scan=r.scan, t=t, id=r.id, point=point, n=n, range_m=r.w.range_m, n_cells=r.w.n_cells,
                        width_m=r.w.width_m, depth_m=r.w.depth_m, jump_m=jump, x=z[0], y=z[1], ex=z[0] - r.true_x, ey=z[1] - r.true_y,
                        vx=v[0], vy=v[1], raw_vx=raw[0], raw_vy=raw[1], true_vx=r.true_vx, true_vy=r.true_vy))
    return pd.DataFrame(out)


def study(samples: pd.DataFrame, noise: MeasurementNoise = MeasurementNoise(), cfg: StudyConfig = StudyConfig()):
    """(series of every hazard and point, summary DataFrame with a row per hazard and point)."""
    series, rows = [], []
    for hid, g in samples.groupby("id"):
        g = g.sort_values("t")
        label = str(g["label"].iloc[0])
        for point in POINTS:
            s = run_filter(g, point, noise, cfg)
            series.append(s)
            c = s[s["n"] > cfg.burn_in]
            if len(c) < cfg.min_samples:
                continue
            true_speed = float(np.hypot(c["true_vx"], c["true_vy"]).mean())
            speed = np.hypot(c["vx"], c["vy"])
            verr = np.hypot(c["vx"] - c["true_vx"], c["vy"] - c["true_vy"])
            raw = c.dropna(subset=["raw_vx"])
            rerr = np.hypot(raw["raw_vx"] - raw["true_vx"], raw["raw_vy"] - raw["true_vy"])
            bias = np.array([c["ex"].mean(), c["ey"].mean()])
            sd = float(np.sqrt(c["ex"].var(ddof=0) + c["ey"].var(ddof=0)))
            static = true_speed < cfg.static_speed_mps
            rows.append(dict(id=int(hid), label=label, point=point, n=len(c), true_speed=true_speed, static=static,
                             phantom_mean=float(speed.mean()) if static else np.nan,
                             phantom_rms=float(np.sqrt((speed ** 2).mean())) if static else np.nan,
                             phantom_max=float(speed.max()) if static else np.nan,
                             vel_rms=float(np.sqrt((verr ** 2).mean())),
                             speed_rms=float(np.sqrt(((speed - np.hypot(c["true_vx"], c["true_vy"])) ** 2).mean())),
                             raw_rms=float(np.sqrt((rerr ** 2).mean())) if len(rerr) else np.nan,
                             bias=float(np.linalg.norm(bias)), offset_sd=sd))
    return (pd.concat(series, ignore_index=True) if series else pd.DataFrame()), pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Per point: the mean over the hazards that stand still (phantom speed) and over the ones that move (velocity error)."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for point, g in summary.groupby("point"):
        st, mv = g[g["static"]], g[~g["static"]]
        rows.append(dict(point=point,
                         static_phantom_rms=float(st["phantom_rms"].mean()) if len(st) else np.nan,
                         static_phantom_max=float(st["phantom_max"].max()) if len(st) else np.nan,
                         static_offset_sd=float(st["offset_sd"].mean()) if len(st) else np.nan,
                         mover_vel_rms=float(mv["vel_rms"].mean()) if len(mv) else np.nan,
                         mover_raw_rms=float(mv["raw_rms"].mean()) if len(mv) else np.nan,
                         bias_mean=float(g["bias"].mean())))
    return pd.DataFrame(rows).set_index("point").loc[[p for p in POINTS if p in set(summary["point"])]].reset_index()


def spikes(series: pd.DataFrame, point: str = "centre", min_error_mps: float = 1.5, last: int = 4) -> pd.DataFrame:
    """The measurements behind the big velocity errors of one point: where the filter velocity is more than `min_error_mps`
    from the truth, and the last `last` measurements of every hazard. Columns: scan, range, cells, size of the visible
    cluster, the jump of the measured point since the previous measurement (jump_m) and its error against the true centre."""
    s = series[series["point"] == point].copy()
    s["vel_err"] = np.hypot(s["vx"] - s["true_vx"], s["vy"] - s["true_vy"])
    tail = s.groupby("id").tail(last).index
    pick = s[(s["vel_err"] > min_error_mps) | s.index.isin(tail)]
    cols = ["id", "scan", "t", "range_m", "n_cells", "width_m", "depth_m", "jump_m", "ex", "ey", "vel_err"]
    return pick[cols].round(2)