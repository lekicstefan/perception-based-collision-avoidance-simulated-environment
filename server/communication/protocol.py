"""Wire format of docs/protocol.md (schema version 1). All values are little-endian.

Unity to Python: an 88-byte header (HEADER) followed by a payload.
Python to Unity: a 12-byte header (COMMAND_HEADER) followed by a payload.
The parse_* functions validate everything and raise ProtocolError, so a malformed message can never
reach the perception code. The build_* functions exist for tests, the replay tool and a fake Unity.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

VERSION = 1
MAGIC_SENSOR = b"AVPS"    # Unity to Python
MAGIC_COMMAND = b"AVPC"   # Python to Unity

PORT_SESSION, PORT_COMMAND, PORT_LIDAR, PORT_CAMERA, PORT_POSE, PORT_ORACLE = 5555, 5556, 5557, 5558, 5559, 5560

# magic, version, type, flags, seq, payload_length, t, x y z yaw pitch roll, speed yaw_rate steering, reserved
HEADER = struct.Struct("<4sHBBIId6d3fI")
COMMAND_HEADER = struct.Struct("<4sHBBI")
# id, data_timestamp, comfort_cap, safety_cap, comfort_decel, brake_request, validity, fallback_decel,
# processing_time, emergency, risk_level, reserved
COMMAND_PAYLOAD = struct.Struct("<IdfffffffBBH")
LIDAR_PREFIX = struct.Struct("<HH")           # rows, cols
CAMERA_PREFIX = struct.Struct("<HHB3x")       # width, height, format, 3 reserved bytes
ORACLE_COUNT = struct.Struct("<I")
ORACLE_DTYPE = np.dtype([("id", "<i4"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("yaw", "<f8"),
                         ("vx", "<f8"), ("vy", "<f8"), ("length", "<f4"), ("width", "<f4"), ("height", "<f4")])

assert HEADER.size == 88 and COMMAND_HEADER.size == 12 and COMMAND_PAYLOAD.size == 44
assert LIDAR_PREFIX.size == 4 and CAMERA_PREFIX.size == 8 and ORACLE_DTYPE.itemsize == 64


class MsgType(IntEnum):
    HELLO = 1
    CALIBRATION = 2
    END_OF_RUN = 3
    LIDAR = 4
    CAMERA = 5
    POSE = 6
    ORACLE = 7


class CmdType(IntEnum):
    READY = 1
    COMMAND = 2
    TELEMETRY = 3


class CameraFormat(IntEnum):
    JPEG = 0
    RAW = 1


SESSION_TYPES = (MsgType.HELLO, MsgType.CALIBRATION, MsgType.END_OF_RUN)


class ProtocolError(ValueError):
    """A message that does not follow docs/protocol.md."""


@dataclass(frozen=True)
class Header:
    type: MsgType
    seq: int
    payload_length: int
    t: float                 # capture time (s)
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    speed: float
    yaw_rate: float
    steering_angle: float

    @property
    def pose(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.yaw, self.pitch, self.roll)


@dataclass(frozen=True)
class CameraPayload:
    width: int
    height: int
    format: CameraFormat
    data: memoryview


@dataclass(frozen=True)
class Command:
    id: int
    data_timestamp: float
    comfort_cap: float
    safety_cap: float
    comfort_decel: float
    brake_request: float
    validity: float
    fallback_decel: float
    processing_time: float
    emergency: bool
    risk_level: int


# ---------------------------------------------------------------- Unity to Python: parsing

def parse_header(msg) -> Header:
    if len(msg) < HEADER.size:
        raise ProtocolError(f"message too short for a header ({len(msg)} bytes)")
    (magic, version, mtype, _flags, seq, plen, t, x, y, z, yaw, pitch, roll,
     speed, yaw_rate, steering, _reserved) = HEADER.unpack_from(msg)
    if magic != MAGIC_SENSOR:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"schema version {version}, expected {VERSION}")
    try:
        mtype = MsgType(mtype)
    except ValueError:
        raise ProtocolError(f"unknown message type {mtype}") from None
    if len(msg) != HEADER.size + plen:
        raise ProtocolError(f"length mismatch: {len(msg)} bytes, header says {HEADER.size + plen}")
    return Header(mtype, seq, plen, t, x, y, z, yaw, pitch, roll, speed, yaw_rate, steering)


def _expect(h: Header, mtype: MsgType) -> None:
    if h.type != mtype:
        raise ProtocolError(f"expected {mtype.name}, got {h.type.name}")


def parse_pose(msg) -> Header:
    h = parse_header(msg)
    _expect(h, MsgType.POSE)
    if h.payload_length != 0:
        raise ProtocolError("POSE has no payload")
    return h


def parse_lidar(msg) -> tuple[Header, np.ndarray]:
    """Returns the header and the range image as a read-only uint16 array (rows, cols), no copy."""
    h = parse_header(msg)
    _expect(h, MsgType.LIDAR)
    if h.payload_length < LIDAR_PREFIX.size:
        raise ProtocolError("LIDAR payload too short")
    rows, cols = LIDAR_PREFIX.unpack_from(msg, HEADER.size)
    if h.payload_length != LIDAR_PREFIX.size + 2 * rows * cols:
        raise ProtocolError(f"LIDAR payload {h.payload_length} bytes does not fit {rows} x {cols}")
    ranges = np.frombuffer(msg, dtype="<u2", count=rows * cols, offset=HEADER.size + LIDAR_PREFIX.size)
    return h, ranges.reshape(rows, cols)


def parse_camera(msg) -> tuple[Header, CameraPayload]:
    h = parse_header(msg)
    _expect(h, MsgType.CAMERA)
    if h.payload_length < CAMERA_PREFIX.size:
        raise ProtocolError("CAMERA payload too short")
    width, height, fmt = CAMERA_PREFIX.unpack_from(msg, HEADER.size)
    try:
        fmt = CameraFormat(fmt)
    except ValueError:
        raise ProtocolError(f"unknown camera format {fmt}") from None
    data = memoryview(msg)[HEADER.size + CAMERA_PREFIX.size:]
    if fmt == CameraFormat.RAW and len(data) != width * height * 3:
        raise ProtocolError(f"raw image of {len(data)} bytes does not fit {width} x {height}")
    return h, CameraPayload(width, height, fmt, data)


def camera_to_rgb(cam: CameraPayload) -> np.ndarray:
    """Decodes to an RGB uint8 array (height, width, 3), top row first. JPEG needs OpenCV."""
    if cam.format == CameraFormat.RAW:
        return np.frombuffer(cam.data, dtype=np.uint8).reshape(cam.height, cam.width, 3)
    import cv2
    bgr = cv2.imdecode(np.frombuffer(cam.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[:2] != (cam.height, cam.width):
        raise ProtocolError("JPEG decode failed or the size does not match the header")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_session(msg) -> tuple[Header, dict]:
    """HELLO, CALIBRATION or END_OF_RUN: the header and the decoded JSON payload."""
    h = parse_header(msg)
    if h.type not in SESSION_TYPES:
        raise ProtocolError(f"{h.type.name} is not a session message")
    try:
        data = json.loads(bytes(memoryview(msg)[HEADER.size:]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad JSON payload: {e}") from None
    if not isinstance(data, dict):
        raise ProtocolError("JSON payload must be an object")
    return h, data


def parse_oracle(msg) -> tuple[Header, np.ndarray]:
    """Oracle mode only. Returns the header and a structured array with one row per object."""
    h = parse_header(msg)
    _expect(h, MsgType.ORACLE)
    if h.payload_length < ORACLE_COUNT.size:
        raise ProtocolError("ORACLE payload too short")
    (n,) = ORACLE_COUNT.unpack_from(msg, HEADER.size)
    if h.payload_length != ORACLE_COUNT.size + n * ORACLE_DTYPE.itemsize:
        raise ProtocolError(f"ORACLE payload {h.payload_length} bytes does not fit {n} objects")
    return h, np.frombuffer(msg, dtype=ORACLE_DTYPE, count=n, offset=HEADER.size + ORACLE_COUNT.size)


# ------------------------------------------------ Unity to Python: building (tests, replay, fake Unity)

def build_message(mtype: MsgType, seq: int, payload: bytes = b"", t: float = 0.0,
                  pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), speed: float = 0.0, yaw_rate: float = 0.0,
                  steering_angle: float = 0.0) -> bytes:
    x, y, z, yaw, pitch, roll = pose
    return HEADER.pack(MAGIC_SENSOR, VERSION, int(mtype), 0, seq, len(payload), t, x, y, z, yaw, pitch, roll,
                       speed, yaw_rate, steering_angle, 0) + payload


def lidar_payload(ranges: np.ndarray) -> bytes:
    rows, cols = ranges.shape
    return LIDAR_PREFIX.pack(rows, cols) + np.ascontiguousarray(ranges, dtype="<u2").tobytes()


def camera_payload(width: int, height: int, fmt: CameraFormat, data: bytes) -> bytes:
    return CAMERA_PREFIX.pack(width, height, int(fmt)) + bytes(data)


def json_payload(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def oracle_payload(objects: np.ndarray) -> bytes:
    objects = np.ascontiguousarray(objects, dtype=ORACLE_DTYPE)
    return ORACLE_COUNT.pack(len(objects)) + objects.tobytes()


# ------------------------------------------------------------ Python to Unity: building and parsing

def build_ready(run_dir: str | None = None) -> bytes:
    payload = json_payload({"run_dir": run_dir}) if run_dir else b""
    return COMMAND_HEADER.pack(MAGIC_COMMAND, VERSION, int(CmdType.READY), 0, len(payload)) + payload


def build_command(c: Command) -> bytes:
    payload = COMMAND_PAYLOAD.pack(c.id, c.data_timestamp, c.comfort_cap, c.safety_cap, c.comfort_decel,
                                   c.brake_request, c.validity, c.fallback_decel, c.processing_time,
                                   1 if c.emergency else 0, c.risk_level, 0)
    return COMMAND_HEADER.pack(MAGIC_COMMAND, VERSION, int(CmdType.COMMAND), 0, len(payload)) + payload


def build_telemetry(obj: dict) -> bytes:
    payload = json_payload(obj)
    return COMMAND_HEADER.pack(MAGIC_COMMAND, VERSION, int(CmdType.TELEMETRY), 0, len(payload)) + payload


def parse_command_message(msg) -> tuple[CmdType, object]:
    """For tests and a fake Unity: READY gives None, COMMAND a Command, TELEMETRY a dict."""
    if len(msg) < COMMAND_HEADER.size:
        raise ProtocolError("message too short for a command header")
    magic, version, ctype, _reserved, plen = COMMAND_HEADER.unpack_from(msg)
    if magic != MAGIC_COMMAND:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"schema version {version}, expected {VERSION}")
    if len(msg) != COMMAND_HEADER.size + plen:
        raise ProtocolError("length mismatch")
    try:
        ctype = CmdType(ctype)
    except ValueError:
        raise ProtocolError(f"unknown command message type {ctype}") from None
    if ctype == CmdType.READY:
        if plen == 0:
            return ctype, None
        try:
            return ctype, json.loads(bytes(memoryview(msg)[COMMAND_HEADER.size:]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ProtocolError(f"bad JSON payload: {e}") from None
    if ctype == CmdType.COMMAND:
        if plen != COMMAND_PAYLOAD.size:
            raise ProtocolError("COMMAND payload has the wrong size")
        (cid, ts, comf, safe, cdec, brk, val, fb, proc, emg, risk, _r) = COMMAND_PAYLOAD.unpack_from(msg, COMMAND_HEADER.size)
        return ctype, Command(cid, ts, comf, safe, cdec, brk, val, fb, proc, bool(emg), risk)
    try:
        return ctype, json.loads(bytes(memoryview(msg)[COMMAND_HEADER.size:]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad JSON payload: {e}") from None