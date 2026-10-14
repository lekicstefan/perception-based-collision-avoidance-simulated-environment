"""Cause logging for every cap and braking decision (step 7.12; section 5.6 and 5.10).

`CauseLog.record` writes one record per cycle, as a dict and (optionally) as one JSON line: the data time, the speed, the risk level, the raw and
the output caps, the braking request, the emergency flag, and the CAUSE of the comfort cap, of the safety cap and of the braking decision (see
decision.Cause: a track id or a virtual occlusion hazard, with state, existence, distance and the first time of arrival). When the smoothing holds a
cap below its raw value, the cause is the one that set it before (`*_held`); when the emergency is latched without a new request, the brake cause
is the one that started it (`latched`). A number that is not finite is written as null.

`reaction_events` turns the records of a run into events for the evaluation, which joins them with the ground truth to assign the reaction classes
(realized-hazard, precautionary, unjustified): a "cap" event is a stretch in which the comfort cap is below the speed of the car (the car is
being slowed by a cap), a "brake" event one in which a braking request or the emergency is active; records of the same type and cause that are at
most `gap_s` apart form one event.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from server.risk.decision import NO_CAUSE, Cause, Decision


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


class CauseLog:
    def __init__(self, path: str | Path | None = None, keep: bool = True):
        self.records = []
        self.keep = keep
        self._f = open(path, "w", encoding="utf-8") if path else None
        self._comfort = self._safety = self._brake = None

    def record(self, d: Decision, *, comfort_cap: float | None = None, safety_cap: float | None = None, comfort_held: bool = False,
               safety_held: bool = False, brake_request: float | None = None, emergency: bool | None = None, latched: bool = False,
               command_id: int | None = None) -> dict:
        comfort = self._comfort if comfort_held and self._comfort is not None else d.comfort_cause
        safety = self._safety if safety_held and self._safety is not None else d.safety_cause
        brake = d.brake_cause
        if latched and brake.kind == "none" and self._brake is not None:
            brake = self._brake
        if not (comfort_held and self._comfort is not None):
            self._comfort = d.comfort_cause
        if not (safety_held and self._safety is not None):
            self._safety = d.safety_cause
        if brake.kind != "none":
            self._brake = brake
        rec = {
            "t": d.t, "command_id": command_id, "speed": d.speed_mps, "risk_level": d.risk_level, "regime": int(d.regime),
            "raw_comfort_cap": _num(d.comfort_cap), "raw_safety_cap": _num(d.safety_cap),
            "comfort_cap": _num(d.comfort_cap if comfort_cap is None else comfort_cap), "safety_cap": _num(d.safety_cap if safety_cap is None else safety_cap),
            "comfort_held": bool(comfort_held), "safety_held": bool(safety_held),
            "brake_request": _num(d.brake_request if brake_request is None else brake_request),
            "emergency": bool(d.emergency if emergency is None else emergency), "latched": bool(latched),
            "n_hazards": d.n_hazards, "n_virtual": d.n_virtual,
            "comfort_cause": comfort.to_dict(), "safety_cause": safety.to_dict(), "brake_cause": brake.to_dict(),
        }
        if self.keep:
            self.records.append(rec)
        if self._f:
            self._f.write(json.dumps(rec) + "\n")
        return rec

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


def read_log(path: str | Path) -> list:
    """The records of a log file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reaction_events(records: list, gap_s: float = 0.35, margin_mps: float = 0.1) -> list:
    """Events {type, label, kind, source_id, t_start, t_end, n, regime_max, peak_brake, min_cap} sorted by their start (see the module text)."""
    open_events, done = {}, []

    def add(typ: str, rec: dict, cause: dict) -> None:
        key = (typ, cause["label"])
        ev = open_events.get(key)
        if ev is not None and rec["t"] - ev["t_end"] > gap_s:
            done.append(ev)
            ev = None
        if ev is None:
            ev = {"type": typ, "label": cause["label"], "kind": cause["kind"], "source_id": cause["source_id"], "t_start": rec["t"],
                  "t_end": rec["t"], "n": 0, "regime_max": 0, "peak_brake": 0.0, "min_cap": None}
            open_events[key] = ev
        ev["t_end"], ev["n"] = rec["t"], ev["n"] + 1
        ev["regime_max"] = max(ev["regime_max"], rec["regime"])
        ev["peak_brake"] = max(ev["peak_brake"], rec["brake_request"] or 0.0)
        cap = rec["comfort_cap"]
        if cap is not None:
            ev["min_cap"] = cap if ev["min_cap"] is None else min(ev["min_cap"], cap)

    for rec in records:
        cap = rec["comfort_cap"]
        if cap is not None and cap < rec["speed"] - margin_mps:
            add("cap", rec, rec["comfort_cause"])
        if (rec["brake_request"] or 0.0) > 0.0 or rec["emergency"]:
            add("brake", rec, rec["brake_cause"])
    done.extend(open_events.values())
    return sorted(done, key=lambda e: (e["t_start"], e["type"], e["label"]))