"""Tests for range image ingestion (steps 5.1 and 5.2)."""
import math

import numpy as np
import pytest

from server.communication import protocol as P
from server.geometry.calibration import Calibration, LidarCalibration, Mount
from server.perception.range_image import CellState, RangeImageBuilder
from testing.support.dev_recording import load_lidar
from testing.support.paths import RECORDINGS_DIR

DEV = RECORDINGS_DIR / "dev_default_seed12345"


def make_lidar(pitch_deg=0.0):
    """3 beams (+4, 0, -4 degrees), 5 columns (+2 ... -2 degrees), sensor at x = 1.0 m, z = 1.8 m."""
    return LidarCalibration(
        rows=3, cols=5, rate_hz=10.0, max_range_m=100.0, mount=Mount(1.0, 0.0, 1.8, 0.0, pitch_deg, 0.0),
        elevations_deg=np.array([4.0, 0.0, -4.0]), azimuths_deg=np.array([2.0, 1.0, 0.0, -1.0, -2.0]))


def message(raw, seq=5, t=1.25, pose=(1.0, 2.0, 0.1, 0.3, 0.0, 0.0), speed=12.5):
    return P.build_message(P.MsgType.LIDAR, seq, P.lidar_payload(raw), t=t, pose=pose, speed=speed,
                           yaw_rate=0.05, steering_angle=-0.02)


def test_the_three_cell_classes_stay_distinct():
    raw = np.array([[0, 65535, 5000, 12345, 1],
                    [5000, 5000, 0, 0, 65535],
                    [100, 200, 300, 400, 500]], dtype=np.uint16)
    img = RangeImageBuilder(make_lidar()).ingest(message(raw))
    assert img.shape == (3, 5)
    assert img.state[0, 0] == CellState.NO_RETURN and img.state[0, 1] == CellState.MAX_RANGE
    assert img.state[0, 2] == CellState.VALID and img.state[0, 4] == CellState.VALID     # even 1 cm is a return
    assert img.range_m[0, 2] == pytest.approx(50.0) and img.range_m[0, 3] == pytest.approx(123.45)
    assert img.range_m[0, 4] == pytest.approx(0.01)
    assert np.isnan(img.range_m[0, 0]) and np.isnan(img.range_m[0, 1])                   # never a distance
    assert img.counts() == {"no_return": 3, "valid": 10, "max_range": 2}
    assert np.array_equal(img.valid, img.state == CellState.VALID)


def test_header_fields_are_carried_over():
    img = RangeImageBuilder(make_lidar()).ingest(message(np.full((3, 5), 5000, dtype=np.uint16)))
    assert img.seq == 5 and img.t == 1.25 and img.pose == (1.0, 2.0, 0.1, 0.3, 0.0, 0.0)
    assert img.speed == 12.5 and img.yaw_rate == pytest.approx(0.05, abs=1e-7)
    assert img.steering_angle == pytest.approx(-0.02, abs=1e-7)


def test_an_image_that_does_not_match_the_calibration_is_rejected():
    with pytest.raises(P.ProtocolError):
        RangeImageBuilder(make_lidar()).ingest(message(np.zeros((4, 5), dtype=np.uint16)))


def test_row_and_column_mapping():
    d = RangeImageBuilder(make_lidar()).geometry.dirs_vehicle
    assert d[0, 2, 2] > 0 > d[2, 2, 2]                    # row 0 is the highest beam
    assert d[1, 0, 1] > 0 > d[1, 4, 1]                    # column 0 is the leftmost (y positive = left)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)


def test_a_point_straight_ahead():
    b = RangeImageBuilder(make_lidar())
    raw = np.zeros((3, 5), dtype=np.uint16)
    raw[1, 2] = 1000                                      # 10 m, elevation 0, azimuth 0
    p = b.geometry.points_vehicle(b.ingest(message(raw)))
    assert p[1, 2] == pytest.approx([11.0, 0.0, 1.8])     # sensor x + 10, sensor height
    assert np.isnan(p[0, 0]).all()                        # cells without a return have no point


def test_a_beam_that_hits_flat_ground():
    b = RangeImageBuilder(make_lidar())
    r = 1.8 / math.sin(math.radians(4.0))                 # slant range at which the -4 degree beam meets the ground
    raw = np.zeros((3, 5), dtype=np.uint16)
    raw[2, 2] = round(r * 100)
    p = b.geometry.points_vehicle(b.ingest(message(raw)))[2, 2]
    assert p[2] == pytest.approx(0.0, abs=0.01)
    assert p[0] == pytest.approx(1.0 + 1.8 / math.tan(math.radians(4.0)), abs=0.02)


def test_the_mount_pitch_is_applied():
    d = RangeImageBuilder(make_lidar(pitch_deg=10.0)).geometry.dirs_vehicle[1, 2]      # elevation 0, azimuth 0
    assert d == pytest.approx([math.cos(math.radians(10)), 0.0, math.sin(math.radians(10))])


def test_angles_between_neighbouring_rays():
    g = RangeImageBuilder(make_lidar()).geometry
    assert g.alpha_h.shape == (3, 4) and g.alpha_v.shape == (2, 5)
    assert g.alpha_v == pytest.approx(math.radians(4.0))                               # 4 degrees between beams
    assert g.alpha_h[1] == pytest.approx(math.radians(1.0))                            # at elevation 0
    assert g.alpha_h[0, 0] == pytest.approx(math.radians(1.0) * math.cos(math.radians(4.0)), rel=1e-3)   # shrinks off the horizon


@pytest.mark.skipif(not (DEV / "lidar.bin").exists(), reason="no development recording in testing/data/recordings")
def test_a_recorded_scan():
    calib = Calibration.from_json(DEV / "calibration.json")
    rec = load_lidar(DEV)[0]
    b = RangeImageBuilder(calib.lidar)
    img = b.ingest(message(rec["ranges"], seq=int(rec["frame"]), t=float(rec["t"]), pose=tuple(rec["pose"])))
    assert img.shape == (calib.lidar.rows, calib.lidar.cols)
    assert sum(img.counts().values()) == img.range_m.size
    assert np.nanmax(img.range_m) <= calib.lidar.max_range_m + 0.02
    assert b.geometry.points_vehicle(img).shape == (calib.lidar.rows, calib.lidar.cols, 3)