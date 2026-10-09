"""Tests for protocol.py against the byte-exact test vectors in docs/protocol.md section 8."""
import numpy as np
import pytest

from server.communication import protocol as P

POSE_VECTOR = bytes.fromhex(
    "41565053010006000700000000000000000000000000f83f0000000000802440000000000000e0bf"
    "9a9999999999b93f9a9999999999b93f0000000000000000000000000000000000004841cdcc4c3d0ad7a33c00000000")
COMMAND_VECTOR = bytes.fromhex(
    "41565043010002002c0000002a000000ae47e17a14aef73f33334b4100008c41000040400000000000002040"
    "cdcc4c3ea69b443c00010000")
COMMAND = P.Command(id=42, data_timestamp=1.48, comfort_cap=12.7, safety_cap=17.5, comfort_decel=3.0,
                    brake_request=0.0, validity=2.5, fallback_decel=0.2, processing_time=0.012,
                    emergency=False, risk_level=1)


def pose_message(seq=7):
    return P.build_message(P.MsgType.POSE, seq, t=1.5, pose=(10.25, -0.5, 0.1, 0.1, 0.0, 0.0),
                           speed=12.5, yaw_rate=0.05, steering_angle=0.02)


def test_sizes():
    assert P.HEADER.size == 88 and P.COMMAND_HEADER.size == 12 and P.COMMAND_PAYLOAD.size == 44
    assert len(POSE_VECTOR) == 88 and len(COMMAND_VECTOR) == 56


def test_pose_vector_is_reproduced():
    assert pose_message() == POSE_VECTOR


def test_pose_vector_is_parsed():
    h = P.parse_pose(POSE_VECTOR)
    assert h.type == P.MsgType.POSE and h.seq == 7 and h.payload_length == 0
    assert h.t == 1.5 and h.pose == (10.25, -0.5, 0.1, 0.1, 0.0, 0.0)
    assert h.speed == 12.5 and h.yaw_rate == pytest.approx(0.05, abs=1e-7)
    assert h.steering_angle == pytest.approx(0.02, abs=1e-7)


def test_command_vector_is_reproduced_and_parsed():
    assert P.build_command(COMMAND) == COMMAND_VECTOR
    ctype, c = P.parse_command_message(COMMAND_VECTOR)
    assert ctype == P.CmdType.COMMAND and c.id == 42 and c.emergency is False and c.risk_level == 1
    assert c.data_timestamp == 1.48
    for name in ("comfort_cap", "safety_cap", "comfort_decel", "brake_request", "validity",
                 "fallback_decel", "processing_time"):
        assert getattr(c, name) == pytest.approx(getattr(COMMAND, name), rel=1e-6)


def test_ready_and_telemetry_round_trip():
    assert P.parse_command_message(P.build_ready()) == (P.CmdType.READY, None)
    ctype, data = P.parse_command_message(P.build_telemetry({"speed": 12.5, "tracks": []}))
    assert ctype == P.CmdType.TELEMETRY and data == {"speed": 12.5, "tracks": []}


def test_lidar_round_trip_and_size():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 65536, size=(32, 600), dtype=np.uint16)
    msg = P.build_message(P.MsgType.LIDAR, 3, P.lidar_payload(img), t=0.3)
    assert len(msg) == 38492                      # 88 + 4 + 2 * 32 * 600
    h, out = P.parse_lidar(msg)
    assert h.seq == 3 and h.t == 0.3 and out.shape == (32, 600) and out.dtype == np.dtype("<u2")
    assert np.array_equal(out, img)
    assert not out.flags.writeable                # a view of the message, no copy


def test_camera_raw_round_trip():
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    msg = P.build_message(P.MsgType.CAMERA, 0, P.camera_payload(6, 4, P.CameraFormat.RAW, img.tobytes()))
    h, cam = P.parse_camera(msg)
    assert (cam.width, cam.height, cam.format) == (6, 4, P.CameraFormat.RAW)
    assert np.array_equal(P.camera_to_rgb(cam), img)


def test_camera_jpeg_round_trip():
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((36, 64, 3), dtype=np.uint8)
    img[:, :32] = (255, 0, 0)                     # red on the left in RGB
    ok, jpg = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    msg = P.build_message(P.MsgType.CAMERA, 1, P.camera_payload(64, 36, P.CameraFormat.JPEG, jpg.tobytes()))
    _, cam = P.parse_camera(msg)
    rgb = P.camera_to_rgb(cam)
    assert rgb.shape == (36, 64, 3)
    assert rgb[18, 5, 0] > 200 and rgb[18, 5, 2] < 60 and rgb[18, 60, 0] < 60   # colours not swapped


def test_session_messages():
    msg = P.build_message(P.MsgType.HELLO, 0, P.json_payload({"schema": 1}))
    h, data = P.parse_session(msg)
    assert h.type == P.MsgType.HELLO and data == {"schema": 1} and h.t == 0.0 and h.pose == (0.0,) * 6
    _, data = P.parse_session(P.build_message(P.MsgType.END_OF_RUN, 2, P.json_payload({"sim_time": 12.34})))
    assert data["sim_time"] == 12.34


def test_oracle_round_trip():
    objs = np.zeros(3, dtype=P.ORACLE_DTYPE)
    objs["id"] = [1, 2, 3]
    objs["x"] = [10.0, 20.0, 30.0]
    objs["vy"] = [0.0, 1.4, -1.4]
    objs["length"] = [4.2, 0.5, 0.5]
    msg = P.build_message(P.MsgType.ORACLE, 5, P.oracle_payload(objs), t=0.5)
    assert len(msg) == 88 + 4 + 3 * 64
    h, out = P.parse_oracle(msg)
    assert h.seq == 5 and len(out) == 3 and np.array_equal(out, objs)


def corrupt(msg: bytes, offset: int, value: bytes) -> bytes:
    return msg[:offset] + value + msg[offset + len(value):]


@pytest.mark.parametrize("bad", [
    POSE_VECTOR[:50],                              # truncated header
    corrupt(POSE_VECTOR, 0, b"XXXX"),              # bad magic
    corrupt(POSE_VECTOR, 4, b"\x02\x00"),          # other schema version
    corrupt(POSE_VECTOR, 6, b"\x63"),              # unknown message type
    POSE_VECTOR + b"\x00",                         # longer than the header says
])
def test_bad_messages_are_rejected(bad):
    with pytest.raises(P.ProtocolError):
        P.parse_header(bad)


def test_wrong_type_and_inconsistent_payload_are_rejected():
    with pytest.raises(P.ProtocolError):
        P.parse_lidar(POSE_VECTOR)
    bad = P.build_message(P.MsgType.LIDAR, 0, P.LIDAR_PREFIX.pack(32, 600) + b"\x00" * 10)
    with pytest.raises(P.ProtocolError):
        P.parse_lidar(bad)
    with pytest.raises(P.ProtocolError):
        P.parse_command_message(corrupt(COMMAND_VECTOR, 0, b"AVPS"))