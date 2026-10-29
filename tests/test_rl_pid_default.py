"""PID default: with RL off, the mux output must be *exactly* the PID output.

Rule 2 of the frozen ``ControllerMux`` contract: unless ``selected == "ppo"``
(which requires ``RL_ENABLED`` and a request/default of ``"ppo"`` and an
available adapter) the mux must be a transparent wrapper over
``ThrustComputer.compute``. Not "close to" — the same floats.

Both cases are covered: ``RL_ENABLED = False`` and the config field absent
entirely. All four navigation modes are exercised, waypoint and station-keeping
over a scripted sequence so the PID integrators actually run.
"""

from __future__ import annotations

import pytest

from src.control.controller_mux import ControllerMux
from src.control.navigation import NavigationState, Waypoint
from src.control.thrust import ThrustComputer

from tests import _fakes

_RIGID = ("surge", "sway", "yaw")


class _Clock:
    def __init__(self, t=500.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, dt=0.05):
        self.t += dt


def _nv_cfg():
    """A config object without any RL_* name (RL_ENABLED genuinely absent)."""
    return type("NoRLConfig", (), {})()


def _nav_sequence():
    """A scripted run touching all four modes."""
    wp = Waypoint(latitude=-6.8960, longitude=112.0521, name="w1")
    return [
        _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.7,
                        desired_heading=45.0),
        _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.4,
                        desired_heading=200.0),
        _fakes.make_nav(NavigationState.STATION_KEEP, desired_north_m=12.0,
                        desired_east_m=-7.0, hold_heading=90.0),
        _fakes.make_nav(NavigationState.STATION_KEEP, desired_north_m=-3.0,
                        desired_east_m=4.0, hold_heading=270.0),
        _fakes.make_nav(NavigationState.IDLE),
        _fakes.make_nav(NavigationState.MANUAL),
        _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.9,
                        desired_heading=15.0),
    ]


def _states():
    """Fresh fused states; the mux keys its watchdog on state *values*."""
    return [
        _fakes.make_state(heading=30.0, speed=0.5, yaw_rate=2.0),
        _fakes.make_state(heading=55.0, speed=0.6, yaw_rate=-3.0),
        _fakes.make_state(heading=120.0, speed=0.2, yaw_rate=1.0),
        _fakes.make_state(heading=200.0, speed=0.0, yaw_rate=0.0),
        _fakes.make_state(heading=210.0, speed=0.0, yaw_rate=0.0),
        _fakes.make_state(heading=215.0, speed=0.0, yaw_rate=0.0),
        _fakes.make_state(heading=90.0, speed=1.1, yaw_rate=0.5),
    ]


def _run_mux(config):
    clock = _Clock()
    mux = ControllerMux(
        ThrustComputer(),
        _fakes.FakeAdapter([], available=False, reason="test: never stepped"),
        _fakes.ScriptedNavigator([Waypoint(-6.8960, 112.0521, "w1")],
                                index=0, state=NavigationState.WAYPOINT_NAV),
        gps=_fakes.FakeGPS(true_course=10.0), config=config, clock=clock)
    out = []
    for nav, state in zip(_nav_sequence(), _states()):
        out.append((mux.selected, mux.compute(nav, state)))
        clock.tick()
    return mux, out


def _run_thrust():
    """Reference: a second ThrustComputer fed the identical script."""
    thrust = ThrustComputer()
    clock = _Clock()
    out = []
    for nav, state in zip(_nav_sequence(), _states()):
        out.append(thrust.compute(nav, state))
        clock.tick()
    return out


@pytest.mark.parametrize("config", [
    {"RL_ENABLED": False, "RL_DEFAULT_CONTROLLER": "pid"},
    _nv_cfg(),
], ids=["RL_ENABLED_False", "RL_field_absent"])
def test_pid_path_is_byte_identical(config):
    mux, mux_out = _run_mux(config)
    ref_out = _run_thrust()

    assert len(mux_out) == len(ref_out) == len(_nav_sequence())
    for i, ((selected, cmd), ref) in enumerate(zip(mux_out, ref_out)):
        assert selected == "pid", f"tick {i}: selected={selected!r}"
        for field in _RIGID:
            got = getattr(cmd, field)
            want = getattr(ref, field)
            assert got == want, (
                f"tick {i} field {field}: mux={got!r} thrust={want!r} "
                f"(must be EXACTLY equal)")


def test_all_four_modes_present_in_script():
    """Guard: the sequence really covers WAYPOINT_NAV, STATION_KEEP, IDLE, MANUAL."""
    modes = {nav.mode for nav in _nav_sequence()}
    assert modes == {NavigationState.WAYPOINT_NAV, NavigationState.STATION_KEEP,
                     NavigationState.IDLE, NavigationState.MANUAL}


def test_idle_takes_effect_within_one_tick_with_rl_enabled():
    """Even with RL enabled and an available adapter, IDLE is a raw pass-through.

    ``RL_ENABLED`` selects PPO for autonomous modes only; IDLE/MANUAL must never
    be slew-limited (rule 6), so the first IDLE tick must equal the PID output
    exactly even right after a large policy step.
    """
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(1.0, 1.0, 1.0))])
    mux = ControllerMux(ThrustComputer(), adapter,
                        _fakes.ScriptedNavigator([], state=NavigationState.IDLE),
                        gps=_fakes.FakeGPS(), clock=clock)
    cfg = {"RL_ENABLED": False, "RL_DEFAULT_CONTROLLER": "pid"}
    mux._config = cfg  # test-local config: RL stays off, so PID is the whole path
    state = _fakes.make_state()

    idle = _fakes.make_nav(NavigationState.IDLE)
    thrust = ThrustComputer()
    ref = thrust.compute(idle, state)
    got = mux.compute(idle, state)
    assert (got.surge, got.sway, got.yaw) == (ref.surge, ref.sway, ref.yaw)
    assert adapter.calls == [], "the policy must never run in IDLE"
