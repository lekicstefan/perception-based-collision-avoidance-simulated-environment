"""The covariance double count switch (HazardConfig.subtract_current_sigma): off by default, same numbers as before; on, the reachable radius only
grows with the position sigma that comes after tau = 0."""
import numpy as np
import pytest

from server.risk.ego_path import EgoGeometry, EgoState, make_corridor
from server.risk.hazards import HazardConfig, find_conflicts, reachable_radius
from server.risk.timing import PredictedTrack

G = EgoGeometry()
CORRIDOR = make_corridor(EgoState(10.0), G)
ON = HazardConfig(subtract_current_sigma=True)
OFF = HazardConfig()


def track(cov):
    corners = np.array([[-0.9, -0.9], [0.9, -0.9], [0.9, 0.9], [-0.9, 0.9]])
    return PredictedTrack(id=1, t=1.0, position=np.array([30.9, 0.0]), velocity=np.zeros(2), cov=np.asarray(cov, float), static=True,
                          velocity_known=True, state="confirmed", existence=1.0, age_s=3.0, surface=np.vstack([corners, [-0.9, 0.0]]))


POS_ONLY = np.diag([0.04, 0.04, 0.0, 0.0])              # position sigma 0.2 m, no velocity uncertainty: the sigma does not grow
WITH_VEL = np.diag([0.04, 0.04, 0.01, 0.01])


def test_it_is_off_by_default():
    assert HazardConfig().subtract_current_sigma is False


def test_off_the_radius_is_as_before():
    tr = track(POS_ONLY)
    assert reachable_radius(tr, 0.0, 2.0, OFF) == pytest.approx(OFF.base_radius_m + 2.0 * 0.2)
    assert reachable_radius(tr, 2.0, 2.0, OFF) == pytest.approx(OFF.base_radius_m + 0.5 * OFF.max_object_accel * 4.0 + 2.0 * 0.2)


def test_on_the_present_sigma_is_not_in_the_radius():
    tr = track(POS_ONLY)
    assert reachable_radius(tr, 0.0, 2.0, ON) == pytest.approx(ON.base_radius_m)
    assert reachable_radius(tr, 2.0, 2.0, ON) == pytest.approx(ON.base_radius_m + 0.5 * ON.max_object_accel * 4.0)   # sigma stays 0.2: no growth


def test_on_the_growth_of_the_sigma_stays_in():
    tr = track(WITH_VEL)
    tau, scale = 2.0, 1.5
    sigma0, sigma_tau = 0.2, float(np.sqrt(0.04 + tau * tau * 0.01))
    assert reachable_radius(tr, tau, scale, ON) == pytest.approx(ON.base_radius_m + 0.5 * ON.max_object_accel * tau * tau + scale * (sigma_tau - sigma0))
    assert reachable_radius(tr, tau, scale, ON) < reachable_radius(tr, tau, scale, OFF)


def test_on_the_conflict_point_moves_towards_the_object_never_away():
    tr = track(WITH_VEL)
    (a,) = find_conflicts([tr], CORRIDOR, 10.0, 0.0, G, OFF, 2.0)
    (b,) = find_conflicts([tr], CORRIDOR, 10.0, 0.0, G, ON, 2.0)
    assert b.s_conflict_m >= a.s_conflict_m and b.d_path_m >= a.d_path_m
    assert b.sigma_d_m == pytest.approx(a.sigma_d_m)         # d_eff still has the whole sigma_d: only the radius changes