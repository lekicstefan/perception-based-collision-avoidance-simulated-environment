"""A stand-in for the Unity side of docs/protocol.md, written with pyzmq.

It binds the same sockets and follows the same lifecycle as Unity's NetworkHost.cs (HELLO until READY, then
CALIBRATION, data streams with per-stream sequence numbers, calibration resend as heartbeat, END_OF_RUN x5).
Used by the tests, and runnable as a script to try the processor without Unity:  python server/fake_unity.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import zmq

import protocol as P

HWM = {"session": 1000, "lidar": 5, "camera": 10, "pose": 100, "oracle": 5}
SOCKET_BUFFER_BYTES = 64 * 1024


class FakeUnity:
    def __init__(self, oracle: bool = False, host: str = "127.0.0.1", ports: dict | None = None,
                 calibration: dict | None = None):
        self.ports = {"session": P.PORT_SESSION, "command": P.PORT_COMMAND, "lidar": P.PORT_LIDAR,
                      "camera": P.PORT_CAMERA, "pose": P.PORT_POSE, "oracle": P.PORT_ORACLE}
        if ports:
            self.ports.update(ports)
        self.ctx = zmq.Context()
        self.pubs: dict[str, zmq.Socket] = {}
        for name in ["session", "lidar", "camera", "pose"] + (["oracle"] if oracle else []):
            s = self.ctx.socket(zmq.PUB)
            s.setsockopt(zmq.SNDHWM, HWM[name])
            if name != "session":
                s.setsockopt(zmq.SNDBUF, SOCKET_BUFFER_BYTES)
            s.setsockopt(zmq.LINGER, 500)
            s.bind(f"tcp://{host}:{self.ports[name]}")
            self.pubs[name] = s
        self.cmd = self.ctx.socket(zmq.SUB)
        self.cmd.setsockopt(zmq.SUBSCRIBE, b"")
        self.cmd.setsockopt(zmq.LINGER, 0)
        self.cmd.connect(f"tcp://{host}:{self.ports['command']}")
        self.seq = {k: 0 for k in ("hello", "calibration", "end_of_run", "lidar", "camera", "pose", "oracle")}
        self.calibration = calibration or {"calibrationId": "fake-calibration"}
        self.ready = False
        self.commands: list = []          # (CmdType, object) received from the processor
        self.sent_calibration_at = -1e9

    # -- helpers
    def _send(self, name: str, mtype: P.MsgType, key: str, payload: bytes = b"", **kw) -> None:
        self.pubs[name].send(P.build_message(mtype, self.seq[key], payload, **kw))
        self.seq[key] += 1

    def poll_commands(self) -> None:
        while True:
            try:
                raw = self.cmd.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            ctype, obj = P.parse_command_message(raw)
            if ctype == P.CmdType.READY:
                self.ready = True
            else:
                self.commands.append((ctype, obj))

    # -- session
    def send_hello(self) -> None:
        self._send("session", P.MsgType.HELLO, "hello", P.json_payload({"schema": P.VERSION}))

    def send_calibration(self) -> None:
        self._send("session", P.MsgType.CALIBRATION, "calibration", P.json_payload(self.calibration))
        self.sent_calibration_at = time.monotonic()

    def heartbeat(self, period: float = 1.0) -> None:
        if time.monotonic() - self.sent_calibration_at >= period:
            self.send_calibration()

    def wait_ready(self, timeout: float = 30.0, hello_period: float = 0.1) -> None:
        """What Unity's net thread does before the run: HELLO every 100 ms until READY, then CALIBRATION."""
        t0 = time.monotonic()
        while not self.ready:
            self.send_hello()
            end = time.monotonic() + hello_period
            while time.monotonic() < end and not self.ready:
                self.poll_commands()
                time.sleep(0.005)
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("no READY from the processor")
        self.send_calibration()

    def end_of_run(self, sim_time: float, copies: int = 5, period: float = 0.1) -> None:
        for _ in range(copies):
            self._send("session", P.MsgType.END_OF_RUN, "end_of_run", P.json_payload({"sim_time": sim_time}))
            time.sleep(period)

    # -- data streams
    def send_pose(self, t: float, pose=(0.0,) * 6, speed: float = 10.0) -> None:
        self._send("pose", P.MsgType.POSE, "pose", t=t, pose=pose, speed=speed)

    def send_lidar(self, t: float, ranges: np.ndarray | None = None, pose=(0.0,) * 6, speed: float = 10.0) -> None:
        if ranges is None:
            ranges = np.full((32, 600), 5000, dtype=np.uint16)        # a flat wall at 50 m
        self._send("lidar", P.MsgType.LIDAR, "lidar", P.lidar_payload(ranges), t=t, pose=pose, speed=speed)

    def send_camera(self, t: float, width: int = 64, height: int = 36, pose=(0.0,) * 6, speed: float = 10.0) -> None:
        data = bytes(width * height * 3)
        self._send("camera", P.MsgType.CAMERA, "camera", P.camera_payload(width, height, P.CameraFormat.RAW, data),
                   t=t, pose=pose, speed=speed)

    def send_oracle(self, t: float, objects: np.ndarray, pose=(0.0,) * 6) -> None:
        self._send("oracle", P.MsgType.ORACLE, "oracle", P.oracle_payload(objects), t=t, pose=pose)

    def run(self, seconds: float, speed: float = 10.0) -> None:
        """Stream at the default rates in real time: pose 50 Hz, camera 30 Hz, LiDAR 10 Hz, heartbeat 1 Hz."""
        t0 = time.monotonic()
        n = 0
        while (t := n * 0.01) < seconds:
            if n % 2 == 0:
                self.send_pose(t, speed=speed)
            if n % 3 == 0:
                self.send_camera(t, speed=speed)
            if n % 10 == 0:
                self.send_lidar(t, speed=speed)
            self.heartbeat()
            self.poll_commands()
            n += 1
            time.sleep(max(0.0, t0 + n * 0.01 - time.monotonic()))

    def close(self) -> None:
        if not self.ctx.closed:
            self.ctx.destroy(linger=500)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    fake = FakeUnity()
    print("fake Unity: sending HELLO until the processor answers READY ...")
    fake.wait_ready()
    print(f"READY received, streaming for {seconds:.0f} s")
    fake.run(seconds)
    fake.end_of_run(seconds)
    print(f"done; commands received: {len(fake.commands)}")
    fake.close()