"""Search of the camera edge parameters against the ground truth (step 6.9, camera part).

The bearing measurement of the tracker looks for the left and right edge of a hazard in the camera image, in a region around
where the track predicts it (server/tracking/bearing.py, edge rule of server/perception/camera_refinement.py). Here the edges it
finds are compared with the TRUE silhouette of the hazard: the left and right pixel column of its box (8 corners projected with
the camera calibration of the recording). For every attempt (track in view of the camera that belongs to a hazard):

  found_rate   share of the attempts with a bearing measurement (both edges found, plausible width)
  good_rate    share with a measurement whose two edges are both within `tol_px` pixels of the true ones
  wrong_rate   share with a measurement that is found but not good (a false edge: the dangerous case, it moves the track)
  rms_mid_px   rms error of the middle of the two edges (the quantity the bearing is made from), of the found ones
  rms_edge_px  rms error of the single edges

    score = (1 - good_rate) + 2 wrong_rate + 0.02 rms_mid_px + 10 static_while_moving_max          (lower is better)

(the last term guards the tracker's safety metric: the camera must not make a mover look static). The parameters (SPACE) are drawn
at random, scored on one or more recordings and the best are refined by small random steps, as in search_study.py.
"""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from server.geometry.calibration import Calibration
from server.perception.camera_refinement import RefinementConfig
from server.tracking.bearing import BearingConfig
from testing.support.fusion_study import replay_fusion
from testing.support.ground_truth import GroundTruth
from testing.support.tuning_study import metrics

# name: (low, high, kind)   kind: f float, i integer, odd odd integer, b boolean (0/1)
SPACE = {
    "canny_low": (10.0, 80.0, "f"),
    "canny_ratio": (1.5, 8.0, "f"),                 # canny_high = canny_low * canny_ratio
    "blur_ksize": (3, 9, "odd"),
    "min_edge_frac": (0.03, 0.7, "f"),
    "min_contrast": (1.0, 12.0, "f"),
    "ambiguity_ratio": (0.4, 0.95, "f"),
    "margin_frac": (0.1, 0.8, "f"),
    "min_margin_px": (2.0, 20.0, "f"),
    "min_width_px": (2.0, 10.0, "f"),
    "width_ratio_lo": (0.2, 0.7, "f"),
    "width_ratio_hi": (1.5, 4.0, "f"),
    "use_color": (0, 1, "b"),
    "color_canny_factor": (0.2, 1.0, "f"),
}
DEFAULT = dict(canny_low=40.0, canny_ratio=3.0, blur_ksize=5, min_edge_frac=0.4, min_contrast=6.0, ambiguity_ratio=0.7, margin_frac=0.3,
               min_margin_px=8.0, min_width_px=4.0, width_ratio_lo=0.4, width_ratio_hi=3.0, use_color=1, color_canny_factor=0.6)
# the set adopted in the first search (BearingConfig defaults since then); DEFAULT above stays the baseline of the comparison
TUNED = dict(canny_low=41.0, canny_ratio=188.0 / 41.0, blur_ksize=5, min_edge_frac=0.10, min_contrast=6.5, ambiguity_ratio=0.68,
             margin_frac=0.48, min_margin_px=4.5, min_width_px=4.2, width_ratio_lo=0.48, width_ratio_hi=3.2, use_color=1,
             color_canny_factor=0.39)
METRICS = ["attempts", "found_rate", "good_rate", "wrong_rate", "no_edge_rate", "ambiguous_rate", "rms_mid_px", "rms_edge_px",
           "vel_rms_moving", "static_while_moving_max"]


def clip(p: dict) -> dict:
    out = {}
    for k, (lo, hi, kind) in SPACE.items():
        v = float(min(max(p[k], lo), hi))
        if kind == "odd":
            v = int(2 * round((v - 1) / 2) + 1)
            v = int(min(max(v, lo), hi))
        elif kind in ("i", "b"):
            v = int(round(v))
        out[k] = v
    return out


def sample(rng: np.random.Generator, n: int) -> list:
    cols = {k: lo + (rng.permutation(n) + rng.random(n)) / n * (hi - lo) for k, (lo, hi, _) in SPACE.items()}
    return [clip({k: cols[k][i] for k in SPACE}) for i in range(n)]


def perturb(rng: np.random.Generator, p: dict, scale: float) -> dict:
    q = {k: p[k] + rng.normal(0.0, scale * (hi - lo)) for k, (lo, hi, _) in SPACE.items()}
    if rng.random() < 0.15:                              # flip the colour edges now and then
        q["use_color"] = 1 - p["use_color"]
    return clip(q)


def build_bearing(p: dict, base: BearingConfig = BearingConfig()) -> BearingConfig:
    edges = replace(base.edges, blur_ksize=int(p["blur_ksize"]), canny_low=float(p["canny_low"]),
                    canny_high=float(p["canny_low"] * p["canny_ratio"]), min_edge_frac=float(p["min_edge_frac"]),
                    min_contrast=float(p["min_contrast"]), ambiguity_ratio=float(p["ambiguity_ratio"]))
    return replace(base, margin_frac=float(p["margin_frac"]), min_margin_px=float(p["min_margin_px"]), min_width_px=float(p["min_width_px"]),
                   width_ratio=(float(p["width_ratio_lo"]), float(p["width_ratio_hi"])), use_color=bool(p["use_color"]),
                   color_canny_factor=float(p["color_canny_factor"]), edges=edges)


def silhouette(obj, camera):
    """(u_left, u_right) pixel columns of the box of a TruthObject (vehicle frame) or None when a corner is not in front."""
    fp = obj.footprint()
    pts = np.array([[x, y, z] for x, y in fp for z in (0.0, obj.height)])
    u, _, depth = camera.project(pts)
    if not (depth > 0.5).all():
        return None
    return float(u.min()), float(u.max())


def edge_metrics(folder, scans: list, bearing: BearingConfig, camera_every: int = 3, burn_in: int = 10, tol_px: float = 5.0) -> dict:
    folder = Path(folder)
    truth = GroundTruth.load(folder)
    camera = Calibration.from_json(folder / "calibration.json").camera
    state = dict(attempts=0, found=0, good=0, reasons=Counter(), mid=[], edge=[])

    def on_camera(t, rgb, pose, diag, results, tracker):
        sil = [s for s in (silhouette(o, camera) for o in truth.objects_at(t)) if s is not None]
        for m in diag:
            if m.expected_u is None or m.reason in ("behind_camera", "image_border") or not sil:
                continue
            mid = 0.5 * (m.expected_u[0] + m.expected_u[1])
            width = m.expected_u[1] - m.expected_u[0]
            j = int(np.argmin([abs(0.5 * (a + b) - mid) for a, b in sil]))
            a, b = sil[j]
            if abs(0.5 * (a + b) - mid) > max(15.0, 0.6 * width + 0.5 * (b - a)):          # a track that belongs to no visible hazard
                continue
            state["attempts"] += 1
            state["reasons"][m.reason] += 1
            if m.reason != "ok" or m.u is None:
                continue
            state["found"] += 1
            e_lo, e_hi = m.u[0] - a, m.u[1] - b
            state["edge"] += [e_lo, e_hi]
            state["mid"].append(0.5 * (e_lo + e_hi))
            state["good"] += int(abs(e_lo) <= tol_px and abs(e_hi) <= tol_px)

    run = replay_fusion(folder, scans, True, bearing=bearing, camera_every=camera_every, burn_in=burn_in, on_camera=on_camera)
    n = max(state["attempts"], 1)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")
    m = metrics(run)
    return dict(attempts=state["attempts"], found_rate=state["found"] / n if state["attempts"] else float("nan"),
                good_rate=state["good"] / n if state["attempts"] else float("nan"),
                wrong_rate=(state["found"] - state["good"]) / n if state["attempts"] else float("nan"),
                no_edge_rate=state["reasons"]["no_edge"] / n if state["attempts"] else float("nan"),
                ambiguous_rate=state["reasons"]["ambiguous"] / n if state["attempts"] else float("nan"),
                rms_mid_px=rms(state["mid"]), rms_edge_px=rms(state["edge"]), vel_rms_moving=m["vel_rms_moving"],
                static_while_moving_max=m["static_while_moving_max"])


def _z(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def score(m: dict) -> float:
    return ((1.0 - _z(m["good_rate"])) + 2.0 * _z(m["wrong_rate"]) + 0.02 * min(_z(m["rms_mid_px"]), 50.0)
            + 10.0 * _z(m["static_while_moving_max"]))


_CONTEXT: list = []


def _init(context):
    global _CONTEXT
    _CONTEXT = context


def _evaluate(args):
    p, camera_every, burn_in, tol_px = args
    return [edge_metrics(folder, scans, build_bearing(p), camera_every, burn_in, tol_px) for folder, scans in _CONTEXT]


class Evaluator:
    def __init__(self, context: list, workers: int = 1, camera_every: int = 3, burn_in: int = 10, tol_px: float = 5.0):
        self.context, self.args = context, (camera_every, burn_in, tol_px)
        self.pool = ProcessPoolExecutor(workers, initializer=_init, initargs=(context,)) if workers > 1 else None
        _init(context)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def evaluate(self, params: list) -> list:
        jobs = [(p, *self.args) for p in params]
        return list(self.pool.map(_evaluate, jobs)) if self.pool is not None else [_evaluate(j) for j in jobs]


def search(ev: Evaluator, n_random: int, rounds: int, per_round: int, top: int, seed: int = 0, log=print) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()

    def run(batch, stage):
        for p, res in zip(batch, ev.evaluate(batch)):
            rows.append(dict(p, stage=stage, score=float(np.mean([score(m) for m in res])),
                             **{k: float(np.nanmean([m[k] for m in res])) for k in METRICS}))
        log(f"  {stage}: {len(rows)} evaluated, best score {min(r['score'] for r in rows):.3f}, {time.time() - t0:.0f} s")

    run([dict(DEFAULT)] + sample(rng, n_random), "random")
    for r in range(rounds):
        scale = 0.15 * (0.2 ** (r / max(rounds - 1, 1)))
        best = sorted(rows, key=lambda x: x["score"])[:top]
        run([perturb(rng, {k: b[k] for k in SPACE}, scale) for b in best for _ in range(per_round)], f"refine {r + 1}")
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def sensitivity(table: pd.DataFrame, top_fraction: float = 0.1) -> pd.DataFrame:
    best = table.head(max(int(len(table) * top_fraction), 3))
    return pd.DataFrame([dict(parameter=k, best_min=float(best[k].min()), best_median=float(best[k].median()), best_max=float(best[k].max()),
                              range_used=float((best[k].max() - best[k].min()) / (hi - lo))) for k, (lo, hi, _) in SPACE.items()])