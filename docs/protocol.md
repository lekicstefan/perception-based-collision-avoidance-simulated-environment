# Communication protocol

Source of truth for every message between the Unity simulator and the Python processor. Both implementations (C# and Python) must follow this document.

## 1. Overview

ZeroMQ over TCP on localhost. NetMQ in Unity (Mono backend, `AsyncIO.ForceDotNet.Force()` on the network thread before creating sockets), pyzmq in Python. One message = one ZeroMQ frame (single-part). Everything is little-endian. All floats are IEEE 754.
Unity-to-Python data is split over several PUB sockets, one per stream, so that a backlog in one stream (for example camera frames) can never delay or displace another (pose, LiDAR, control). Python-to-Unity traffic uses one PUB socket.

| Socket | Bound by | Port | Direction | Carries | Rate | Send HWM (Unity) / receive HWM (Python) |
|---|---|---|---|---|---|---|
| session | Unity | 5555 | Unity to Python | HELLO, CALIBRATION, END_OF_RUN | low | 1000 |
| command | Python | 5556 | Python to Unity | READY, COMMAND, TELEMETRY | 30 Hz + 10 Hz | 10 |
| lidar | Unity | 5557 | Unity to Python | LIDAR | 10 Hz | 5 |
| camera | Unity | 5558 | Unity to Python | CAMERA | 30 Hz | 10 |
| pose | Unity | 5559 | Unity to Python | POSE | 50 Hz | 100 |
| oracle | Unity | 5560 | Unity to Python | ORACLE | 10 Hz | 5 |

- The **oracle socket exists only in oracle mode.** In normal mode Unity never binds port 5560 and Python never connects to it. The information-boundary test relies on this.
- HWM values are initial. Set the send HWM on the Unity PUB socket **and** the receive HWM on the Python SUB socket to the same value, because the effective queue is the sum of both sides plus kernel buffers.
- A PUB socket that reaches its HWM drops **new** messages, not old ones. A stalled processor therefore sees stale data first, then a gap. The gap is detected through the per-stream sequence number and counted in the run log.
- `ZMQ_CONFLATE` is deliberately not used on the sensor streams: the tracker needs every measurement, and Python drains the queues only once per cycle.
- Subscribers subscribe to everything (empty prefix). The first byte(s) of a message are therefore the magic number, not a topic.

## 2. Data conventions

- **Timestamp** `t`: simulation time of the capture step, float64 seconds. Sensors are stamped with the fixed step nearest to their ideal sample time (phase 3). Within one stream timestamps never decrease. Streams are independent; Python merges them by timestamp.
- **Frame of reference:** processor convention, right-handed, x forward, y left, z up. Pose is relative to the pose at the start of the run (start frame). Angles in radians, yaw positive left, pitch positive up, yaw wrapped to (-pi, pi].
- **Pose noise:** when enabled in the simulator, the noise is already inside the pose fields. The processor cannot tell and receives no flag.
- **Sequence number** (`seq`): per stream (and per message type on the session socket), starts at 0 and increases by exactly 1. Python treats a gap as lost messages, and a first message with `seq != 0` as a start-up loss (slow joiner).
- **Range image:** uint16 centimetres, row-major, row 0 = highest beam, column 0 = leftmost (azimuth positive to the left). `0` = no return (dropout, fog, grazing; NOT free space), `65535` = reached maximum range or sky.
- **Camera image:** RGB, top row first, sRGB. Format 0 = JPEG file bytes, format 1 = raw RGB (3 bytes per pixel, row-major, no padding, `width * height * 3` bytes).

## 3. Unity-to-Python message header (88 bytes)

Every message on the session, lidar, camera, pose and oracle sockets starts with this header. The pose fields are the ego pose **at the capture step of the message**, so image, scan and pose always match.

| Offset | Size | Type | Field | Notes |
|---|---|---|---|---|
| 0 | 4 | char[4] | magic | ASCII `AVPS` |
| 4 | 2 | uint16 | version | schema version, currently 1 |
| 6 | 1 | uint8 | type | see section 4 |
| 7 | 1 | uint8 | flags | reserved, 0 |
| 8 | 4 | uint32 | seq | section 2 |
| 12 | 4 | uint32 | payload_length | bytes after the header |
| 16 | 8 | float64 | t | capture time (s) |
| 24 | 8 | float64 | x | m |
| 32 | 8 | float64 | y | m |
| 40 | 8 | float64 | z | m |
| 48 | 8 | float64 | yaw | rad |
| 56 | 8 | float64 | pitch | rad |
| 64 | 8 | float64 | roll | rad |
| 72 | 4 | float32 | speed | m/s |
| 76 | 4 | float32 | yaw_rate | rad/s, positive left |
| 80 | 4 | float32 | steering_angle | rad, positive left |
| 84 | 4 | uint32 | reserved | 0 (keeps the payload 8-byte aligned) |

Python: `struct.Struct("<4sHBBIId6d3fI")` (size 88). A receiver must check magic, version and `len(message) == 88 + payload_length`, and drop the message with a logged error otherwise. On the session socket, t and the pose fields of HELLO, CALIBRATION and END_OF_RUN are zero.

## 4. Unity-to-Python message types

| Type | Name | Socket | Payload |
|---|---|---|---|
| 1 | HELLO | session | JSON |
| 2 | CALIBRATION | session | JSON |
| 3 | END_OF_RUN | session | JSON |
| 4 | LIDAR | lidar | binary, section 4.1 |
| 5 | CAMERA | camera | binary, section 4.2 |
| 6 | POSE | pose | none (`payload_length = 0`) |
| 7 | ORACLE | oracle | binary, section 4.3 (oracle mode only) |

Rule: rare control messages use UTF-8 JSON payloads (readable, easy to extend), data messages use binary payloads.

### 4.1 LIDAR payload

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 2 | uint16 | rows (32 by default) |
| 2 | 2 | uint16 | cols (600 by default) |
| 4 | 2 * rows * cols | uint16[rows][cols] | range image (section 2) |

`payload_length = 4 + 2 * rows * cols`, which is 38 404 bytes at the defaults (38 492 with the header). Python: `np.frombuffer(msg, dtype="<u2", count=rows*cols, offset=92).reshape(rows, cols)`.

### 4.2 CAMERA payload

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 2 | uint16 | width |
| 2 | 2 | uint16 | height |
| 4 | 1 | uint8 | format (0 = JPEG, 1 = raw RGB) |
| 5 | 3 | uint8[3] | reserved, 0 |
| 8 | n | bytes | image data |

JPEG is the default (decision for the project: decode cost is small on localhost and JPEG quality is configurable, default 92). Raw stays selectable through the sensor configuration. `CameraFrame.latencyMs` is **never** sent.

### 4.3 ORACLE payload (oracle mode only)

This message deliberately breaks the information boundary. Results obtained with it are an upper bound, never "the system". The header carries the same (possibly noisy) pose as the LiDAR scan it belongs to, and `t` equals that scan's timestamp. Object coordinates are the true start-frame values, so use oracle mode with pose noise off.

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 4 | uint32 | n objects |
| 4 | 64 * n | record | objects |

Record (64 bytes): `int32 id`, `float64 x, y, z, yaw, vx, vy` (m, rad, m/s, start frame), `float32 length, width, height` (m). Python: `struct.Struct("<i6d3f")`.

### 4.4 Session messages (JSON payloads)

- **HELLO:** `{"schema": 1}`. Sent by Unity every 100 ms from start-up until READY is received.
- **CALIBRATION:** the `CalibrationData` JSON built by `CalibrationBuilder` (compact form, `CalibrationProvider.Json`), which contains `calibrationId`, the content hash. Sent once right after READY and then every 1 s. A receiver ignores a repeat with the same id. A changed id during a run is an error. It contains no noise, dropout, fog or seed parameters.
- **END_OF_RUN:** `{"sim_time": <float>}`. Nothing else: the reason for the end (collision, route finished, ...) stays in Unity's own log. Sent 5 times, 100 ms apart, followed by a socket linger of 500 ms, so the loss of one copy cannot hide it. Receivers deduplicate.

`seq` on the session socket counts per message type.

## 5. Python-to-Unity messages (command socket)

Header (12 bytes), shared by all three types:

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 4 | char[4] | magic, ASCII `AVPC` |
| 4 | 2 | uint16 | version (1) |
| 6 | 1 | uint8 | type: 1 READY, 2 COMMAND, 3 TELEMETRY |
| 7 | 1 | uint8 | reserved, 0 |
| 8 | 4 | uint32 | payload_length |

Python: `struct.Struct("<4sHBBI")` (size 12).

### 5.1 READY (type 1)

No payload. Sent only after Python has finished its warm-up and connected all of its SUB sockets. Resent every 100 ms until the first POSE message arrives.

### 5.2 COMMAND (type 2), payload 44 bytes (message 56 bytes)

Mirrors Unity's `VehicleCommand` struct.

| Offset | Type | Field | Unit / meaning |
|---|---|---|---|
| 0 | uint32 | id | command id, increasing |
| 4 | float64 | data_timestamp | simulation time of the newest data used |
| 12 | float32 | comfort_cap | m/s, held in normal operation |
| 16 | float32 | safety_cap | m/s, hard limit |
| 20 | float32 | comfort_decel | m/s^2, limit when slowing toward the comfort cap |
| 24 | float32 | brake_request | m/s^2, >= 0, 0 = none |
| 28 | float32 | validity | s, counted from the step Unity applies it |
| 32 | float32 | fallback_decel | m/s^2 after expiry, 0 = release |
| 36 | float32 | processing_time | s, measured by the processor (log only) |
| 40 | uint8 | emergency | 0 or 1 |
| 41 | uint8 | risk_level | 0 low, 1 medium, 2 high (display and log only) |
| 42 | uint16 | reserved | 0 |

Python: `struct.Struct("<IdfffffffBBH")` (size 44). Unity keeps only the newest command (`CommandExecutor.Submit`, the executor already takes the latest one per step).

### 5.3 TELEMETRY (type 3)

UTF-8 JSON payload for the dashboard, about 10 Hz. Never used for control. Unity ignores unknown keys.

## 6. Handshake and lifecycle

1. **Python starts first** (or at any time before the run is meant to begin): binds the command PUB (5556), connects SUB sockets to session, lidar, camera and pose (and oracle only in oracle mode), and runs the Numba warm-up on dummy data.
2. **Unity starts:** binds its PUB sockets, connects a SUB to 5556, and publishes HELLO every 100 ms. The scenario does **not** start yet and no sensor data is published.
3. **Python** receives HELLO, checks `schema`, waits 300 ms (subscriptions settle), then sends READY every 100 ms.
4. **Unity** receives READY: publishes CALIBRATION, starts the scenario, and the sensors start publishing with `seq = 0`.
5. **Python** stops sending READY at the first POSE message. For each stream it checks that the first `seq` is 0 and logs a start-up loss otherwise.
6. **During the run:** CALIBRATION is resent every 1 s from the network thread in wall-clock time, also while the simulation is paused. It doubles as a heartbeat. Python ends the run with a timeout error when no session message arrives for 5 s.
7. **End:** Unity sends END_OF_RUN and shuts down its sockets. Python flushes its logs on the first END_OF_RUN and exits.

A run is one Unity process plus one Python process. If either dies, the run is invalid and is discarded (there is no reconnection logic). If the processor dies mid-run, Unity applies the fallback of the last command when it expires. That is the failure-injection experiment.

## 7. Information boundary

Allowed to reach Python in normal mode: the message types HELLO, CALIBRATION, END_OF_RUN, LIDAR, CAMERA, POSE, with exactly the fields in sections 3 and 4.
Never sent: object lists, true object poses or velocities, scene layout, semantic labels, ground-truth hit ids, the noise-free pose when pose noise is on, simulator internals (noise, dropout, fog or seed parameters, `latencyMs`). ORACLE messages exist only on port 5560 in oracle mode.
The automated test checks: (a) port 5560 is not bound in normal mode, (b) every message type seen is in the allowed list, (c) every message parses with exactly the layouts above, (d) the calibration JSON contains none of the forbidden keys.

## 8. Test vectors

Both implementations must reproduce these bytes exactly (unit test on each side).

**POSE** (`seq = 7`, `t = 1.5`, `x = 10.25`, `y = -0.5`, `z = 0.1`, `yaw = 0.1`, `pitch = 0`, `roll = 0`, `speed = 12.5`, `yaw_rate = 0.05`, `steering = 0.02`, no payload), 88 bytes:

```
41565053010006000700000000000000000000000000f83f0000000000802440000000000000e0bf9a9999999999b93f9a9999999999b93f0000000000000000000000000000000000004841cdcc4c3d0ad7a33c00000000
```

**COMMAND** (`id = 42`, `data_timestamp = 1.48`, `comfort_cap = 12.7`, `safety_cap = 17.5`, `comfort_decel = 3.0`, `brake_request = 0`, `validity = 2.5`, `fallback_decel = 0.2`, `processing_time = 0.012`, `emergency = 0`, `risk_level = 1`), 56 bytes:

```
41565043010002002c0000002a000000ae47e17a14aef73f33334b4100008c41000040400000000000002040cdcc4c3ea69b443c00010000
```

## 9. Versioning and sizes

- `version` is the schema version. Any change to a header or payload layout increments it. The receiver rejects any other version.
- Typical message sizes at the defaults: POSE 88 B, LIDAR 38 492 B, CAMERA 88 B + 8 B + JPEG (roughly 15 to 40 kB), COMMAND 56 B.
- Typical data rate: about 0.4 MB/s LiDAR, about 1 MB/s camera (JPEG), about 4 kB/s pose.

## 10. Implementation locations

Proposed, matching the current repository layout:

- Unity: `Assets/Scripts/Network/Protocol.cs` (constants, header writer), one network thread, one thread-safe queue per outgoing socket.
- Python: `server/protocol.py` (constants, header and payload parsers, command builder), with pytest tests that check the test vectors in section 8.
