import json
import struct

import numpy as np

from recording import camera_path, lidar_dtype, load_lidar, load_meta


def write_record(f, frame, t, ranges):
    f.write(struct.pack("<i", frame))
    f.write(struct.pack("<d", t))
    f.write(struct.pack("<6d", 1.0, 2.0, 3.0, 0.1, 0.2, 0.3))
    f.write(struct.pack("<3f", 4.0, 0.5, -0.25))
    f.write(np.asarray(ranges, dtype="<u2").tobytes())


def test_record_size():
    assert lidar_dtype(32, 600).itemsize == 4 + 8 + 48 + 12 + 32 * 600 * 2


def test_load_lidar_roundtrip(tmp_path):
    rows, cols = 2, 3
    (tmp_path / "meta.json").write_text(json.dumps({"rows": rows, "cols": cols}))
    with open(tmp_path / "lidar.bin", "wb") as f:
        write_record(f, 0, 0.01, [[100, 0, 65535], [250, 1000, 2000]])
        write_record(f, 1, 0.11, [[101, 0, 65535], [251, 1001, 2001]])
    data = load_lidar(tmp_path)
    assert len(data) == 2
    assert load_meta(tmp_path)["rows"] == 2
    assert data["frame"].tolist() == [0, 1]
    assert np.allclose(data["t"], [0.01, 0.11])
    assert np.allclose(data["pose"][0], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert np.allclose(data["motion"][1], [4.0, 0.5, -0.25])
    assert data["ranges"][0, 0, 2] == 65535 and data["ranges"][1, 1, 1] == 1001
    assert data["ranges"].shape == (2, 2, 3)


def test_camera_path(tmp_path):
    assert camera_path(tmp_path, 42).name == "000042.jpg"