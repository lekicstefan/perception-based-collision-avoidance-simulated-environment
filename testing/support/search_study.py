"""Random search and local refinement of the tracker parameters against the ground truth (step 6.9, long run).

The parameters (SPACE) are drawn at random, scored on one or more recordings, and the best ones are refined by small random
steps. The score (lower is better) puts the dangerous errors first:

    score = vel_rms_moving + 0.5 vel_rms_still + 10 static_while_moving_max + 1.5 (1 - static_while_still_mean)
            + 1.0 leave_latency_max_s + 20 max(0, 1 - rows / rows_reference)

i.e. the velocity error, a mover flagged static (worst hazard), a still hazard not flagged static (so that a configuration that
simply never flags anything static does not win), the time to leave the static state, and lost measurements (fewer scored
rows than the default configuration). With several recordings the score is the mean over them. The weights are `Weights`.
Without the camera: the camera bearing hardly changes these numbers (see 6.9) and is tuned separately.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from server.tracking.noise import MeasurementNoise
from server.tracking.outputs import OutputConfig
from testing.support.fusion_study import replay_fusion
from testing.support.tuning_study import metrics

# name: (low, high, integer)
SPACE = {
    "sigma_a": (0.5, 4.0, False),
    "sigma_extent": (0.3, 1.5, False),
    "gate": (6.0, 30.0, False),
    "known_sigma": (0.3, 2.0, False),
    "static_max_speed": (0.3, 1.0, False),
    "static_confirm_scans": (3, 12, True),
    "leave_chi2": (9.2, 20.0, False),
}
DEFAULT = dict(sigma_a=1.5, sigma_extent=0.7, gate=9.21, known_sigma=0.6, static_max_speed=0.6, static_confirm_scans=5, leave_chi2=13.8)


@dataclass(frozen=True)
class Weights:
    vel_moving: float = 1.0
    vel_still: float = 0.5
    static_while_moving: float = 10.0
    static_while_still: float = 1.5
    leave_latency: float = 1.0
    lost_rows: float = 20.0


def clip(params: dict) -> dict:
    out = {}
    for k, (lo, hi, integer) in SPACE.items():
        v = float(min(max(params[k], lo), hi))
        out[k] = int(round(v)) if integer else v
    return out


def sample(rng: np.random.Generator, n: int) -> list:
    """n random parameter sets, stratified (Latin hypercube) in every parameter."""
    cols = {}
    for k, (lo, hi, integer) in SPACE.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        cols[k] = lo + u * (hi - lo)
    return [clip({k: cols[k][i] for k in SPACE}) for i in range(n)]


def perturb(rng: np.random.Generator, params: dict, scale: float) -> dict:
    return clip({k: params[k] + rng.normal(0.0, scale * (hi - lo)) for k, (lo, hi, _) in SPACE.items()})


def run_params(folder, scans, p: dict, burn_in: int = 10) -> dict:
    return replay_fusion(folder, scans, False, gate_chi2=p["gate"], sigma_a=p["sigma_a"], burn_in=burn_in,
                         noise=replace(MeasurementNoise(), sigma_extent_m=p["sigma_extent"]),
                         output=OutputConfig(known_sigma_mps=p["known_sigma"], static_max_speed_mps=p["static_max_speed"],
                                             static_confirm_scans=int(p["static_confirm_scans"]), leave_chi2=p["leave_chi2"]))


def _z(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def score(m: dict, rows_reference: float, w: Weights = Weights()) -> float:
    lost = max(0.0, 1.0 - m["rows"] / rows_reference) if rows_reference > 0 else 0.0
    return (w.vel_moving * _z(m["vel_rms_moving"]) + w.vel_still * _z(m["vel_rms_still"])
            + w.static_while_moving * _z(m["static_while_moving_max"])
            + w.static_while_still * (1.0 - _z(m["static_while_still_mean"])) + w.leave_latency * _z(m["leave_latency_max_s"])
            + w.lost_rows * lost)


_CONTEXT: list = []          # [(folder, scans)] of the worker process


def _init(context):
    global _CONTEXT
    _CONTEXT = context


def _evaluate(args):
    p, burn_in = args
    return [metrics(run_params(folder, scans, p, burn_in)) for folder, scans in _CONTEXT]


class Evaluator:
    """Scores parameter sets on the recordings; in parallel when workers > 1."""

    def __init__(self, context: list, workers: int = 1, burn_in: int = 10, weights: Weights = Weights()):
        self.context, self.burn_in, self.weights = context, burn_in, weights
        self.pool = ProcessPoolExecutor(workers, initializer=_init, initargs=(context,)) if workers > 1 else None
        _init(context)                  # also in this process: the reference below and the serial case use it
        self.reference = [m["rows"] for m in _evaluate((DEFAULT, burn_in))]

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def score_of(self, per_recording: list) -> float:
        return float(np.mean([score(m, ref, self.weights) for m, ref in zip(per_recording, self.reference)]))

    def evaluate(self, params: list) -> list:
        jobs = [(p, self.burn_in) for p in params]
        return list(self.pool.map(_evaluate, jobs)) if self.pool is not None else [_evaluate(j) for j in jobs]


def search(ev: Evaluator, n_random: int, rounds: int, per_round: int, top: int, seed: int = 0, log=print) -> pd.DataFrame:
    """Random search of n_random sets, then `rounds` of refinement: `per_round` perturbations of each of the `top` best,
    with the step shrinking from 15 % to 3 % of the range. Returns every evaluated set (columns of SPACE, score, per-recording
    metrics of the mean), best first."""
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()

    def run(batch, stage):
        for p, res in zip(batch, ev.evaluate(batch)):
            rows.append(dict(p, stage=stage, score=ev.score_of(res), **{k: float(np.nanmean([m[k] for m in res])) for k in res[0] if k != "nis"},
                             worst_recording_score=max(score(m, ref, ev.weights) for m, ref in zip(res, ev.reference))))
        log(f"  {stage}: {len(rows)} evaluated, best score {min(r['score'] for r in rows):.3f}, {time.time() - t0:.0f} s")

    run([dict(DEFAULT)] + sample(rng, n_random), "random")
    for r in range(rounds):
        scale = 0.15 * (0.2 ** (r / max(rounds - 1, 1)))
        best = sorted(rows, key=lambda x: x["score"])[:top]
        run([perturb(rng, {k: b[k] for k in SPACE}, scale) for b in best for _ in range(per_round)], f"refine {r + 1}")
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def sensitivity(table: pd.DataFrame, top_fraction: float = 0.1) -> pd.DataFrame:
    """For every parameter: its range among the best `top_fraction` of the evaluated sets against its full range. A narrow
    range means the score depends strongly on it; a wide one means it hardly matters."""
    n = max(int(len(table) * top_fraction), 3)
    best = table.head(n)
    out = []
    for k, (lo, hi, _) in SPACE.items():
        out.append(dict(parameter=k, best_min=float(best[k].min()), best_median=float(best[k].median()), best_max=float(best[k].max()),
                        range_used=float((best[k].max() - best[k].min()) / (hi - lo))))
    return pd.DataFrame(out)