"""Information-boundary checks (step 4.9, docs/protocol.md section 7).

The processor may only know what a real car knows. In normal mode the stream contains exactly these message types
(HELLO, CALIBRATION, END_OF_RUN, LIDAR, CAMERA, POSE) with exactly the fields of the protocol. This module holds an
independent golden copy of what is allowed and checks messages, whole recordings and the open ports against it.
Adding a field or a message type to the protocol therefore fails the checks until the golden copy here is changed
on purpose.

  python server/boundary.py runs/<id>              # check a recorded run (its python/stream.bin)
  python server/boundary.py runs/<id> --ports      # also check that the oracle port is closed (Unity must be running)
  python server/boundary.py <stream.bin> --oracle  # oracle mode: ORACLE messages are allowed
"""
from __future__ import annotations

import argparse
import math
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

from server.communication import protocol as P
from server.recording.stream_log import iter_records, resolve_stream_path

# ------------------------------------------------------------------ golden copy of what is allowed

ALLOWED_TYPES = frozenset({P.MsgType.HELLO, P.MsgType.CALIBRATION, P.MsgType.END_OF_RUN,
                           P.MsgType.LIDAR, P.MsgType.CAMERA, P.MsgType.POSE})
SESSION_TYPES = frozenset({P.MsgType.HELLO, P.MsgType.CALIBRATION, P.MsgType.END_OF_RUN})

HEADER_FORMAT = "<4sHBBIId6d3fI"
HEADER_FIELDS = ("magic", "version", "type", "flags", "seq", "payload_length", "t", "x", "y", "z", "yaw", "pitch",
                 "roll", "speed", "yaw_rate", "steering_angle", "reserved")
LIDAR_PREFIX_FORMAT, LIDAR_PREFIX_FIELDS = "<HH", ("rows", "cols")
CAMERA_PREFIX_FORMAT, CAMERA_PREFIX_FIELDS = "<HHB3x", ("width", "height", "format")

HELLO_KEYS = frozenset({"schema", "run_dir"})
END_OF_RUN_KEYS = frozenset({"sim_time"})

_MOUNT = ("x", "y", "z", "yawDeg", "pitchDeg", "rollDeg")
ALLOWED_CALIBRATION_PATHS = frozenset(
    ["schema", "calibrationId", "conventions", "poseRateHz", "ego", "camera", "lidar", "camera.mount",
     "camera.opticalFromVehicle", "camera.opticalFromVehicle.R", "camera.opticalFromVehicle.t", "lidar.mount"]
    + [f"ego.{k}" for k in ("wheelbase", "length", "width", "height", "rearOverhang", "frontOverhang")]
    + [f"camera.{k}" for k in ("width", "height", "rateHz", "format", "fx", "fy", "cx", "cy", "horizontalFovDeg",
                               "verticalFovDeg", "distortion")]
    + [f"camera.mount.{k}" for k in _MOUNT]
    + [f"lidar.{k}" for k in ("rows", "cols", "rateHz", "maxRangeM", "elevationsDeg", "azimuthsDeg", "rangeUnitM",
                              "noReturnCode", "maxRangeCode")]
    + [f"lidar.mount.{k}" for k in _MOUNT])

# simulator internals and ground truth must never show up in a key name of a session message
FORBIDDEN_WORDS = ("noise", "dropout", "fog", "seed", "latency", "hazard", "object", "label", "truth", "oracle",
                   "scenario", "semantic")


# ------------------------------------------------------------------ checks

def key_paths(obj, prefix: str = "") -> set[str]:
    """All dotted key paths of a JSON object (lists are transparent)."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.add(p)
            out |= key_paths(v, p)
    elif isinstance(obj, list):
        for v in obj:
            out |= key_paths(v, prefix)
    return out


def check_schema() -> list[str]:
    """The protocol definitions in protocol.py against the golden copy above."""
    problems = []
    if P.HEADER.format != HEADER_FORMAT:
        problems.append(f"header layout changed: {P.HEADER.format!r}, allowed {HEADER_FORMAT!r}")
    elif len(P.HEADER.unpack(bytes(P.HEADER.size))) != len(HEADER_FIELDS):
        problems.append("header has a different number of fields than the allowed list")
    if P.LIDAR_PREFIX.format != LIDAR_PREFIX_FORMAT:
        problems.append(f"LIDAR payload prefix changed: {P.LIDAR_PREFIX.format!r}")
    if P.CAMERA_PREFIX.format != CAMERA_PREFIX_FORMAT:
        problems.append(f"CAMERA payload prefix changed: {P.CAMERA_PREFIX.format!r}")
    extra = set(P.MsgType) - ALLOWED_TYPES - {P.MsgType.ORACLE}
    if extra:
        problems.append("message types not on the allowed list: " + ", ".join(sorted(t.name for t in extra)))
    return problems


def _check_json(mtype: P.MsgType, data: dict) -> list[str]:
    problems = []
    paths = key_paths(data)
    for p in sorted(paths):
        for word in FORBIDDEN_WORDS:
            if word in p.lower():
                problems.append(f"{mtype.name}: key '{p}' contains the forbidden word '{word}'")
    if mtype == P.MsgType.HELLO:
        allowed = HELLO_KEYS
    elif mtype == P.MsgType.END_OF_RUN:
        allowed = END_OF_RUN_KEYS
    else:
        allowed = ALLOWED_CALIBRATION_PATHS
    for p in sorted(paths - allowed):
        problems.append(f"{mtype.name}: key '{p}' is not on the allowed list")
    return problems


def check_message(raw: bytes, normal_mode: bool = True) -> list[str]:
    """Problems of one Unity-to-Python message (an empty list means it respects the boundary)."""
    try:
        h = P.parse_header(raw)
    except P.ProtocolError as e:
        return [f"unparseable message: {e}"]
    if normal_mode and h.type not in ALLOWED_TYPES:
        return [f"{h.type.name} message in normal mode"]
    problems = []
    vals = P.HEADER.unpack_from(raw)
    if vals[3]:
        problems.append(f"{h.type.name}: flags are {vals[3]}, must be 0")
    if vals[16]:
        problems.append(f"{h.type.name}: reserved field is {vals[16]}, must be 0")
    if not all(math.isfinite(v) for v in vals[6:16]):
        problems.append(f"{h.type.name}: a header value is not finite")
    try:
        if h.type == P.MsgType.LIDAR:
            P.parse_lidar(raw)
        elif h.type == P.MsgType.CAMERA:
            P.parse_camera(raw)
        elif h.type == P.MsgType.POSE:
            P.parse_pose(raw)
        elif h.type == P.MsgType.ORACLE:
            P.parse_oracle(raw)
        else:
            if any(vals[6:16]):
                problems.append(f"{h.type.name}: time or pose fields are not zero on a session message")
            problems += _check_json(h.type, P.parse_session(raw)[1])
    except P.ProtocolError as e:
        problems.append(f"{h.type.name}: {e}")
    return problems


@dataclass
class StreamReport:
    messages: int = 0
    counts: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_stream(path, normal_mode: bool = True, max_problems: int = 50) -> StreamReport:
    """Checks every message of a recording (python/stream.bin)."""
    rep = StreamReport()
    for i, (_, raw) in enumerate(iter_records(path)):
        rep.messages += 1
        name = "UNKNOWN"
        if len(raw) > 6:
            try:
                name = P.MsgType(raw[6]).name
            except ValueError:
                pass
        rep.counts[name] = rep.counts.get(name, 0) + 1
        for p in check_message(raw, normal_mode):
            if len(rep.problems) < max_problems:
                rep.problems.append(f"record {i}: {p}")
    return rep


def port_is_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def check_ports(normal_mode: bool = True, host: str = "127.0.0.1", oracle_port: int = P.PORT_ORACLE) -> tuple[list[str], dict]:
    """In normal mode Unity must not have bound the oracle port. Returns (problems, {port name: open})."""
    ports = {"session": P.PORT_SESSION, "lidar": P.PORT_LIDAR, "camera": P.PORT_CAMERA, "pose": P.PORT_POSE,
             "oracle": oracle_port}
    state = {name: port_is_open(port, host) for name, port in ports.items()}
    problems = []
    if normal_mode and state["oracle"]:
        problems.append(f"port {oracle_port} (oracle) is open in normal mode")
    return problems, state


def main() -> int:
    ap = argparse.ArgumentParser(description="Information-boundary check of a recorded run")
    ap.add_argument("path", nargs="?", help="a stream.bin, a run folder or its python folder")
    ap.add_argument("--oracle", action="store_true", help="oracle mode: ORACLE messages are allowed")
    ap.add_argument("--ports", action="store_true", help="also check the open ports (Unity must be running)")
    args = ap.parse_args()
    normal = not args.oracle
    problems = check_schema()
    print("schema:", "OK" if not problems else f"{len(problems)} PROBLEMS")
    if args.path:
        rep = check_stream(resolve_stream_path(args.path), normal)
        print(f"stream: {rep.messages} messages {rep.counts} -> {'OK' if rep.ok else str(len(rep.problems)) + ' PROBLEMS'}")
        problems += rep.problems
    if args.ports:
        port_problems, state = check_ports(normal)
        print("ports open:", {k: v for k, v in state.items()}, "->", "OK" if not port_problems else "PROBLEM")
        problems += port_problems
    for p in problems[:20]:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())