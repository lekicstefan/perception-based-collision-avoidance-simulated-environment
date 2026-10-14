"""The TELEMETRY message for the dashboard (step 7.14): one JSON object per scan, at most 10 Hz. Never used for control (protocol.md 5.3).

`build_telemetry(result, command)` turns the `StageResult` of a scan and its command into a plain dict (schema 1):
  t, command_id, speed
  command     what was sent: comfort_cap, safety_cap, comfort_decel, brake_request, emergency, risk_level, validity, fallback_decel
  raw         the caps and the request before the smoothing
  risk        level (filtered), raw_level, sight_distance_m, sight_limit, unknown_fraction, sight_ratio, cause
  decision    regime, n_hazards, n_virtual, comfort_held, safety_held, latched, and the causes (7.12 Cause dicts) of comfort cap, safety cap, brake
  collision   {p, method, per_track} or null
  tracks      every considered track at the time of effect, in the vehicle frame: id, state, x, y, vx, vy, speed, static, existence, sigma_pos,
              corners (the four corners of its visible box, or null)
  virtual     the virtual hazards: kind, cause, s_m, d_path_m, tau_first_s
  timing      stage_ms (view, hazards, collision, decision, total), latency_s, t_effect, and what the caller adds (the processor: perception_tracking_ms)
Every number is a plain float or int; a value that is not finite is null. Unknown keys must be ignored by the reader.
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = 1


def clean(x):
    """JSON-safe copy: numpy scalars and arrays to Python, tuples to lists, non-finite floats to None, enums to ints."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def _track(tr) -> dict:
    v = np.asarray(tr.velocity, dtype=float)
    p = np.asarray(tr.position, dtype=float)
    corners = None if tr.surface is None else (p + np.asarray(tr.surface, dtype=float)[:4]).tolist()
    return {"id": tr.id, "state": tr.state, "x": p[0], "y": p[1], "vx": v[0], "vy": v[1], "speed": float(np.hypot(*v)),
            "static": tr.static, "existence": tr.existence, "sigma_pos": math.sqrt(max(float(np.linalg.eigvalsh(tr.pos_cov)[-1]), 0.0)),
            "corners": corners}


def build_telemetry(res, command=None, extra_timing: dict | None = None) -> dict:
    c = command if command is not None else res.command
    d, a, m = res.decision, res.assessment, res.measure
    out = {
        "schema": SCHEMA, "t": c.data_timestamp, "command_id": c.id, "speed": d.speed_mps,
        "command": {"comfort_cap": c.comfort_cap, "safety_cap": c.safety_cap, "comfort_decel": c.comfort_decel, "brake_request": c.brake_request,
                    "emergency": c.emergency, "risk_level": c.risk_level, "validity": c.validity, "fallback_decel": c.fallback_decel},
        "raw": {"comfort_cap": d.comfort_cap, "safety_cap": d.safety_cap, "brake_request": d.brake_request, "emergency": d.emergency},
        "risk": {"level": int(a.level), "raw_level": res.raw_level, "sight_distance_m": a.sight_distance_m, "sight_limit": None if m is None else m.limit,
                 "unknown_fraction": a.unknown_fraction, "sight_ratio": a.sight_ratio, "cause": a.cause},
        "decision": {"regime": int(d.regime), "n_hazards": d.n_hazards, "n_virtual": d.n_virtual,
                     "comfort_held": res.record["comfort_held"], "safety_held": res.record["safety_held"], "latched": res.record["latched"],
                     "comfort_cause": res.record["comfort_cause"], "safety_cause": res.record["safety_cause"], "brake_cause": res.record["brake_cause"]},
        "collision": None if res.collision is None else {"p": res.collision.p_total, "method": res.collision.method,
                                                         "per_track": {str(k): v for k, v in res.collision.per_track.items()}},
        "tracks": [_track(tr) for tr in res.tracks],
        "virtual": [{"kind": v.kind, "cause": v.cause, "s_m": v.s_m, "d_path_m": v.d_path_m, "tau_first_s": v.tau_first_s} for v in res.virtuals],
        "timing": {"stage_ms": res.timings_ms, "latency_s": res.latency_s, "t_effect": res.t_effect, **(extra_timing or {})},
    }
    return clean(out)