import math
from pathlib import Path

import numpy as np
import pytest

from calibration import Calibration, Mount, rotation_from_euler

DATA = Path(__file__).parent / "results" / "recorded_data.json"


@pytest.fixture(scope="module")
def cal():
    return Calibration.from_json(DATA)


def test_rotation_known_values():
    r = rotation_from_euler(30, 10, 20)
    assert np.allclose(r[:, 0], [0.8529, 0.4924, 0.1736], atol=1e-4)    # forward axis
    assert np.allclose(r[:, 1], [-0.5213, 0.7841, 0.3368], atol=1e-4)   # left axis
    assert np.allclose(r[:, 2], [0.0297, -0.3778, 0.9254], atol=1e-4)   # up axis


def test_rotation_single_axes():
    assert rotation_from_euler(0, 10, 0)[2, 0] > 0                        # positive pitch: nose up
    assert rotation_from_euler(90, 0, 0)[1, 0] == pytest.approx(1.0)      # positive yaw: forward turns left
    assert rotation_from_euler(0, 0, 90)[1, 2] == pytest.approx(-1.0)     # positive roll: up tilts right


def test_rotation_is_orthonormal():
    r = rotation_from_euler(33, -7, 15)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_calibration_id_and_ego(cal):
    assert len(cal.calibration_id) == 8
    assert cal.ego.wheelbase == pytest.approx(2.7)
    assert 1.5 < cal.ego.width < 2.3
    assert cal.ego.rear_overhang > 0 and cal.ego.front_overhang > 0


def test_camera_intrinsics_match_the_field_of_view(cal):
    c = cal.camera
    assert c.fx == pytest.approx((c.width / 2) / math.tan(math.radians(c.horizontal_fov_deg / 2)), rel=1e-5)
    assert c.fy == pytest.approx(c.fx)
    assert (c.cx, c.cy) == (c.width / 2, c.height / 2)
    vfov = 2 * math.degrees(math.atan((c.height / 2) / c.fy))
    assert c.vertical_fov_deg == pytest.approx(vfov, abs=0.01)


def test_projection_of_the_test_cubes(cal):
    # red cube at Unity (-3, 1, 15) and green cube at (3, 1, 15), seen from a vehicle at the origin facing +Z
    u, v, depth = cal.camera.project(np.array([[15.0, 3.0, 1.0], [15.0, -3.0, 1.0]]))
    assert np.allclose(u, [247.27, 392.73], atol=0.05)
    assert np.allclose(v, [187.27, 187.27], atol=0.05)
    assert np.allclose(depth, 13.2, atol=1e-5)


def test_optical_axis_hits_the_principal_point(cal):
    m = cal.camera.mount
    p = m.translation() + 10.0 * m.rotation()[:, 0]
    u, v, d = cal.camera.project(p)
    assert (u, v, d) == pytest.approx((cal.camera.cx, cal.camera.cy, 10.0), abs=1e-5)


def test_left_goes_left_and_up_goes_up_in_the_image(cal):
    m = cal.camera.mount
    r, t = m.rotation(), m.translation()
    u, v, _ = cal.camera.project(t + 10.0 * r[:, 0] + 1.0 * r[:, 1])    # 1 m to the left
    assert u == pytest.approx(cal.camera.cx - cal.camera.fx / 10.0, abs=1e-5)
    u, v, _ = cal.camera.project(t + 10.0 * r[:, 0] + 1.0 * r[:, 2])    # 1 m up
    assert v == pytest.approx(cal.camera.cy - cal.camera.fy / 10.0, abs=1e-5)


def test_optical_transform_matches_the_mount(cal):
    m = cal.camera.mount
    o = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    r_opt = o @ m.rotation().T
    assert np.allclose(cal.camera.r_optical_from_vehicle, r_opt, atol=1e-9)
    assert np.allclose(cal.camera.t_optical_from_vehicle, -r_opt @ m.translation(), atol=1e-9)


def test_lidar_tables(cal):
    l = cal.lidar
    assert l.elevations_deg.shape == (l.rows,) and l.azimuths_deg.shape == (l.cols,)
    assert np.all(np.diff(l.elevations_deg) < 0)       # row 0 is the highest beam
    assert np.all(np.diff(l.azimuths_deg) < 0)         # column 0 is the leftmost
    assert l.azimuths_deg[0] == pytest.approx(59.9, abs=1e-4)
    assert l.azimuths_deg[-1] == pytest.approx(-59.9, abs=1e-4)


def test_lidar_directions_orientation(cal):
    d = cal.lidar.directions()
    assert d.shape == (cal.lidar.rows, cal.lidar.cols, 3)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)
    assert d[0, 0, 2] > d[-1, 0, 2]          # row 0 points higher than the last row
    assert d[0, 0, 1] > 0 > d[0, -1, 1]      # column 0 is on the left


def test_lidar_ray_in_the_vehicle_frame(cal):
    l = cal.lidar
    row = int(np.argmin(np.abs(l.elevations_deg - 0.2667)))
    col = int(np.argmin(np.abs(l.azimuths_deg - 0.1)))
    p_sensor = 10.0 * l.directions()[row, col]
    p_vehicle = l.mount.to_vehicle(p_sensor)
    assert np.allclose(p_vehicle, [11.3, 0.0175, 1.8465], atol=2e-3)