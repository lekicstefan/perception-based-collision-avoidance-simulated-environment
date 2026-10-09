"""Builds a synthetic recording (python/stream.bin format) for the replay and boundary tests."""
import numpy as np

from server.communication import protocol as P
from server.recording.stream_log import StreamRecorder


def write_recording(path, seconds=1.0):
    """1 s at the default rates: pose 50 Hz, camera 30 Hz, LiDAR 10 Hz, plus calibration and end of run."""
    rec = StreamRecorder(path)
    rec.write(P.build_message(P.MsgType.CALIBRATION, 0, P.json_payload({"calibrationId": "synthetic"})), 0.0)
    seq = {"pose": 0, "camera": 0, "lidar": 0}
    cam = P.camera_payload(4, 2, P.CameraFormat.RAW, bytes(24))
    lid = P.lidar_payload(np.full((4, 8), 5000, dtype=np.uint16))
    for i in range(int(seconds * 100)):
        t = i * 0.01
        if i % 2 == 0:
            rec.write(P.build_message(P.MsgType.POSE, seq["pose"], t=t), t); seq["pose"] += 1
        if i % 3 == 0:
            rec.write(P.build_message(P.MsgType.CAMERA, seq["camera"], cam, t=t), t); seq["camera"] += 1
        if i % 10 == 0:
            rec.write(P.build_message(P.MsgType.LIDAR, seq["lidar"], lid, t=t), t); seq["lidar"] += 1
    rec.write(P.build_message(P.MsgType.END_OF_RUN, 0, P.json_payload({"sim_time": seconds})), seconds)
    rec.close()
    return seq