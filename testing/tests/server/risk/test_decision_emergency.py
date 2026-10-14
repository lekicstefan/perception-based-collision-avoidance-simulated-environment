"""When "no distance left" is an emergency (step 7.15): only when the car cannot stop before the hazard even without margin and covariance."""
import math

from server.risk.decision import decide, physically_unavoidable
from server.risk.hazards import Conflict
from server.risk.risk_level import BUNDLES, RiskLevel
from server.risk.stopping import Regime

P = BUNDLES[RiskLevel.MEDIUM].apply()                                   # delay 0.2 s, margin 2, kappa 1.5, eps 0.02, a_max 8


def conflict(d):
    return Conflict(track_id=1, state="confirmed", static=True, tau_first_s=0.0, s_conflict_m=d + 3.5, d_path_m=d, segment_m=(d + 3.5, d + 4.0),
                    t_clear_s=math.inf, v_along_mps=0.0, v_sigma_mps=0.0, sigma_d_m=0.5, margin_weight=1.0, existence=1.0)


def test_using_up_the_margin_at_a_slow_speed_is_hard_braking_not_an_emergency():
    d = decide(0.0, 1.4, [conflict(3.3)], [], P)                         # d_eff < 0 (margin and covariance), but 3.3 m are plenty at 1.4 m/s
    assert not d.emergency and d.regime == Regime.HARD and d.brake_request == P.max_decel


def test_a_stop_that_the_nominal_distance_allows_is_not_an_emergency_even_at_speed():
    d = decide(0.0, 10.0, [conflict(9.0)], [], P)                        # free 7 m: 100 / 14 = 7.1 m/s^2 <= 8, but the margin is gone
    assert not d.emergency and d.regime == Regime.HARD and d.brake_request == P.max_decel


def test_a_stop_that_the_nominal_distance_does_not_allow_is_an_emergency():
    d = decide(0.0, 12.0, [conflict(8.0)], [], P)                        # free 5.6 m: 144 / 11.2 = 12.9 m/s^2 > 8
    assert d.emergency and d.regime == Regime.EMERGENCY and d.brake_request == P.max_decel


def test_no_distance_left_at_all_is_an_emergency_above_the_creeping_speed():
    assert decide(0.0, 5.0, [conflict(0.05)], [], P).emergency
    assert not decide(0.0, 0.6, [conflict(0.05)], [], P).emergency        # creeping: harmless
    assert physically_unavoidable(decide(0.0, 5.0, [conflict(0.05)], [], P).responses[0], 5.0, P)