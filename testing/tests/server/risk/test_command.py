"""The command of a cycle (step 7.13)."""
import math

import pytest

from server.communication import protocol as P
from server.risk.command import CommandConfig, build_command
from server.risk.stopping import StoppingParams

PARAMS = StoppingParams(comfort_decel=2.5)


def cmd(**kw):
    a = dict(command_id=1, data_t=5.0, comfort_cap=10.0, safety_cap=12.0, brake_request=0.0, emergency=False, risk_level=1, params=PARAMS)
    a.update(kw)
    return build_command(**a)


def test_a_plain_command_carries_its_values():
    c = cmd()
    assert (c.id, c.data_timestamp, c.comfort_cap, c.safety_cap, c.brake_request, c.emergency, c.risk_level) == (1, 5.0, 10.0, 12.0, 0.0, False, 1)
    assert c.comfort_decel == 2.5 and c.fallback_decel == 2.5                      # the fallback is the a_comf of the level
    assert c.validity == pytest.approx(0.25)


def test_the_comfort_cap_is_never_above_the_safety_cap():
    c = cmd(comfort_cap=12.0, safety_cap=8.0)
    assert c.comfort_cap == 8.0 and c.safety_cap == 8.0


def test_caps_are_finite_and_not_negative():
    assert cmd(comfort_cap=math.inf, safety_cap=math.inf).safety_cap == 40.0
    c = cmd(comfort_cap=math.nan, safety_cap=-3.0)
    assert c.comfort_cap == 0.0 and c.safety_cap == 0.0
    assert cmd(comfort_cap=math.nan, safety_cap=math.nan).safety_cap == 0.0        # no number is no license to drive


def test_the_braking_request_is_limited_and_an_emergency_asks_for_the_maximum():
    assert cmd(brake_request=3.0).brake_request == 3.0
    assert cmd(brake_request=-1.0).brake_request == 0.0
    assert cmd(brake_request=20.0).brake_request == 8.0
    c = cmd(brake_request=1.0, emergency=True)
    assert c.emergency and c.brake_request == 8.0


def test_the_fallback_and_the_validity_can_be_set():
    c = cmd(cfg=CommandConfig(validity_periods=3.0, command_period_s=0.1, fallback_decel=0.0))
    assert c.fallback_decel == 0.0 and c.validity == pytest.approx(0.3)
    assert cmd(cfg=CommandConfig(fallback_decel=50.0)).fallback_decel == 8.0


def test_the_command_survives_the_wire():
    c = cmd(command_id=7, data_t=12.5, emergency=True, risk_level=2)
    kind, back = P.parse_command_message(P.build_command(c))
    assert kind == P.CmdType.COMMAND
    assert (back.id, back.data_timestamp, back.emergency, back.risk_level) == (7, 12.5, True, 2)
    assert back.comfort_cap == pytest.approx(c.comfort_cap) and back.validity == pytest.approx(0.25)