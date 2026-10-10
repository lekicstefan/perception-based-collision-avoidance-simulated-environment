"""Runs the perception pipeline over a development recording and scores it against the ground truth."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from server.geometry.calibration import Calibration
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from testing.support.ground_truth import GroundTruth
from testing.support.perception_eval import EvalConfig, evaluate_scan
from testing.support.replay_scans import run_pipeline


def evaluate_recording(folder, config: PerceptionConfig, eval_cfg: EvalConfig = EvalConfig(), scans=None,
                       use_camera: bool = False):
    """(DataFrame with a row per hazard and scan, DataFrame with a row per cluster and scan)."""
    folder = Path(folder)
    pipeline = PerceptionPipeline(Calibration.from_json(folder / "calibration.json"), config)
    truth = GroundTruth.load(folder)
    sensor_xy = pipeline.builder.geometry.origin[:2]
    haz_rows, cl_rows = [], []
    for k, rec, result in run_pipeline(folder, pipeline, use_camera=use_camera, scans=scans):
        h, c = evaluate_scan(k, float(rec["t"]), result.clusters, truth.objects_at(float(rec["t"])), sensor_xy, eval_cfg)
        haz_rows += h
        cl_rows += c
    return pd.DataFrame(haz_rows), pd.DataFrame(cl_rows)