"""T10 — fault ladder matrix for ``ControllerMux``.

One test per row of the frozen ladder in ``p1_interfaces.md`` §ControllerMux rule 4
(plus the orchestrator's new ``no_goal`` row). Every row asserts the exact
``(source, fault)`` pair, a finite command inside ``[-1, 1]``, and that no emitted
command steps by more than ``rate_limit_per_s * policy_dt_s`` — except the
``IDLE``/``MANUAL`` pass-through, which by rule 6 must never be slew-limited.

Attribute names are read from the real ``src/control/controller_mux.py``:
``mux.source``, ``mux.fault``, ``mux.selected``, ``mux._latched``; config is the
real ``src.config`` with only ``RL_ENABLED`` raised so the PPO path is reachable.

The mux and adapter share one clock object here (a fake), because ``RLAdapter``'s
vendored watchdog compares the mux-supplied ``est["t"]`` against its own clock —
a fake on one side only stamps every step ``fault="watchdog"``.
"""

from __future__ import annotations

import math

import pytest

from src.control.controller_mux import ControllerMux
from src.control.navigation import NavigationState, Waypoint
from src.control.thrust import ThrustComputer

from tests import _fakes

# rate_limit_per_s (2.0, from FakeAdapter.safety) * policy_dt_s (0.1)
MAX_STEP = 0.2
_EPS = 1e-9

_FAULT_MODES = (NavigationState.WAYPOINT_NAV, NavigationState.STATION_KEEP)


class _Clock:
    """Fake monotonic clock — passed to the mux *and* (implicitly) shared."""

    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, dt=0.1):
        self.t += dt


def _cfg(**over):
    """Dict config: only RL_ENABLED raised; everything else from src.config."""
    from src import config

    base = {
        "RL_ENABLED": True,
        "RL_DEFAULT_CONTROLLER": "pid",
        "RL_GEOFENCE_M": config.RL_GEOFENCE_M,
        "RL_WATCHDOG_S": config.RL_WATCHDOG_S,
        "RL_MAX_INFERENCE_MS": config.RL_MAX_INFERENCE_MS,
        "RL_MAX_THRUST_SCALE": config.RL_MAX_THRUST_SCALE,
    }
    base.update(over)
    return base


def _mux(adapter=None, navigator=None, clock=None, cfg=None, gps=None):
    """A mux wired for the PPO path, with one item queued so ``step`` is due."""
    clock = clock or _Clock()
    if adapter is None:
        adapter = _fakes.FakeAdapter(
            [_fakes.PolicyStep(action=(0.9, 0.0, 0.0))], policy_dt_s=0.1,
            safety={"rate_limit_per_s": 2.0})
    if navigator is None:
        navigator = _fakes.ScriptedNavigator(
            [Waypoint(-6.8969, 112.0521, "w1")], index=0,
            state=NavigationState.WAYPOINT_NAV)
    mux = ControllerMux(ThrustComputer(), adapter, navigator,
                        gps=gps or _fakes.FakeGPS(true_course=12.0),
                        config=cfg or _cfg(), clock=clock)
    mux.set_controller_request("ppo")
    return mux


def _finite_in_range(cmd):
    """Rule: every emitted command is finite and inside [-1, 1] on all axes."""
    vals = (cmd.surge, cmd.sway, cmd.yaw)
    assert all(math.isfinite(v) for v in vals), f"non-finite command {vals!r}"
    assert all(-1.0 - _EPS <= v <= 1.0 + _EPS for v in vals), \
        f"command out of [-1,1]: {vals!r}"
    return vals


def _assert_pair(mux, source, fault):
    assert mux.source == source, f"source={mux.source!r}, want {source!r}"
    assert mux.fault == fault, f"fault={mux.fault!r}, want {fault!r}"


def _assert_no_big_step(commands):
    """Consecutive emitted commands may not step by > rate_limit * dt."""
    for i in range(1, len(commands)):
        for axis in ("surge", "sway", "yaw"):
            step = abs(getattr(commands[i], axis) - getattr(commands[i - 1], axis))
            assert step <= MAX_STEP + _EPS, (
                f"tick {i} axis {axis}: step {step:.6f} > {MAX_STEP} "
                "(rate_limit_per_s * policy_dt_s)")


# --------------------------------------------------------------------------- 1..4
@pytest.mark.parametrize("field, fault", [
    ("gps_valid", "gps"),
    ("imu_valid", "imu"),
    ("mag_valid", "mag"),
])
def test_sensor_invalid_rows(field, fault):
    """gps / imu / mag invalid -> fallback/<field>, PID for the tick."""
    clock = _Clock()
    mux = _mux(clock=clock)
    state = _fakes.make_state(**{field: False})
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.6,
                          desired_heading=30.0)

    cmd = mux.compute(nav, state)
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", fault)
    assert mux.selected == "ppo", "adapter is available, selection must be ppo"
    assert mux.adapter.calls == [], "policy must not run on a sensor fault"


def test_stale_sensor_watchdog():
    """A frozen sample older than RL_WATCHDOG_S -> fallback/'watchdog'.

    The mux keys freshness on state *values* (including ``timestamp``), so the
    stale samples must pin ``timestamp`` explicitly — ``make_state()`` stamps a
    fresh ``time.time()`` per call, which would read as a changing sample.
    """
    clock = _Clock()
    cfg = _cfg(RL_WATCHDOG_S=0.5)
    mux = _mux(clock=clock, cfg=cfg)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)
    frozen = _fakes.make_state(heading=10.0, timestamp=1234.0)

    mux.compute(nav, frozen)         # records freshness at t0, step 1
    clock.tick(0.1)
    cmd = mux.compute(nav, frozen)   # identical values -> still fresh (0.1 s)
    assert mux.fault in (None, "watchdog"), f"unexpected early fault {mux.fault!r}"

    clock.tick(1.0)                  # now > RL_WATCHDOG_S since the last change
    cmd = mux.compute(nav, frozen)
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "watchdog")
    assert len(mux.adapter.calls) == 2, \
        "the policy ran on the two fresh ticks and never after the sample went stale"

    # A *changed* sample restores freshness.
    clock.tick(0.1)
    cmd = mux.compute(nav, _fakes.make_state(heading=99.0, timestamp=1235.0))
    _finite_in_range(cmd)
    assert mux.fault != "watchdog", "a changing sample must clear the watchdog"


# --------------------------------------------------------------------------- 5..7
def test_adapter_step_raises_inference():
    """``adapter.step`` raising -> fallback/'inference'."""
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(raises=RuntimeError("boom"))])
    mux = _mux(adapter=adapter, clock=clock)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.4)

    cmd = mux.compute(nav, _fakes.make_state(heading=5.0))
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "inference")


def test_non_finite_action_nan():
    """A non-finite policy action -> fallback/'nan' (never breaks the tick)."""
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(float("nan"), 0.1, 0.0))])
    mux = _mux(adapter=adapter, clock=clock)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.4)

    cmd = mux.compute(nav, _fakes.make_state(heading=5.0))
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "nan")


def test_slow_inference_over_budget():
    """``last_inference_ms > RL_MAX_INFERENCE_MS`` -> fallback/'slow_inference'."""
    clock = _Clock()
    cfg = _cfg(RL_MAX_INFERENCE_MS=50.0)
    adapter = _fakes.FakeAdapter(
        [_fakes.PolicyStep(action=(1.0, 0.0, 0.0), inference_ms=999.0)])
    mux = _mux(adapter=adapter, clock=clock, cfg=cfg)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.4)

    cmd = mux.compute(nav, _fakes.make_state(heading=5.0))
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "slow_inference")


# ----------------------------------------------------------------------------- 8
def test_three_consecutive_faults_latch_until_new_command():
    """3 consecutive ladder faults -> latched PID reporting fault='latched'."""
    clock = _Clock()
    mux = _mux(clock=clock)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)
    bad = _fakes.make_state(gps_valid=False)

    seen = []
    for _ in range(3):
        seen.append((mux.compute(nav, bad), mux.source, mux.fault))
        clock.tick(0.1)  # budget the tick but the same *values* are re-fed

    assert seen[0][1:] == ("fallback", "gps"), f"fault 1: {seen[0][1:]}"
    assert seen[1][1:] == ("fallback", "gps"), f"fault 2: {seen[1][1:]}"
    assert seen[2][1:] == ("fallback", "latched"), f"fault 3: {seen[2][1:]}"
    assert mux._latched is True

    # Latched survives a perfectly good state.
    good = _fakes.make_state(heading=42.0)
    cmd = mux.compute(nav, good)
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "latched")
    assert mux.adapter.calls == [], "latched mux must not run the policy"

    # A new operator command clears it.
    mux.set_controller_request("ppo")
    clock.tick(0.1)
    cmd = mux.compute(nav, _fakes.make_state(heading=43.0))
    _finite_in_range(cmd)
    _assert_pair(mux, "policy", None)
    assert mux._latched is False

    # A fresh nav command via reset() also clears it.
    mux.set_controller_request("ppo")
    for _ in range(3):
        mux.compute(nav, bad)
        clock.tick(0.1)
    assert mux._latched is True
    mux.reset()
    clock.tick(0.1)
    cmd = mux.compute(nav, _fakes.make_state(heading=44.0))
    _finite_in_range(cmd)
    assert mux._latched is False


# ----------------------------------------------------------------------------- 9
def test_geofence_breach_zero_plus_stop_plus_latch():
    """Beyond RL_GEOFENCE_M -> source='zero', navigator.stop(), latched."""
    clock = _Clock()
    cfg = _cfg(RL_GEOFENCE_M=20.0)
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(1.0, 0.0, 0.0))])
    navigator = _fakes.ScriptedNavigator(
        [Waypoint(-6.8969, 112.0521, "w1")], index=0,
        state=NavigationState.WAYPOINT_NAV)
    mux = _mux(adapter=adapter, navigator=navigator, clock=clock, cfg=cfg)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)

    # Tick 1: the first valid fix of the RL session establishes the ENU origin.
    base = _fakes.make_state(latitude=-6.8969, longitude=112.0521, heading=1.0)
    cmd = mux.compute(nav, base)
    _finite_in_range(cmd)
    assert mux._lat0 == pytest.approx(-6.8969)
    _assert_pair(mux, "policy", None)

    # Tick 2: ~56 m north of the origin -> beyond the 20 m mux geofence.
    clock.tick(0.1)
    moved = _fakes.make_state(latitude=-6.8969 + 0.0005, longitude=112.0521,
                              heading=2.0)
    cmd = mux.compute(nav, moved)
    _finite_in_range(cmd)
    _assert_pair(mux, "zero", "geofence")
    assert (cmd.surge, cmd.sway, cmd.yaw) == (0.0, 0.0, 0.0), "must be zero thrust"
    assert navigator.stopped == 1, "navigator.stop() must be called exactly once"
    assert navigator.state == NavigationState.IDLE
    assert mux._latched is True
    assert len(adapter.calls) == 1, "the policy must not run on a breached tick"

    # Latched until a new command.
    clock.tick(0.1)
    cmd = mux.compute(nav, _fakes.make_state(heading=1.0))
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "latched")

    mux.set_controller_request("ppo")
    clock.tick(0.1)
    cmd = mux.compute(nav, _fakes.make_state(latitude=-6.8969, longitude=112.0521,
                                             heading=2.0))
    _finite_in_range(cmd)
    assert mux._latched is False
    _assert_pair(mux, "policy", None)


# ---------------------------------------------------------------------------- 10
def test_adapter_unavailable_while_ppo_requested_latches():
    """Unavailable adapter + RL enabled + ppo requested -> 'unavailable', latched."""
    clock = _Clock()
    adapter = _fakes.FakeAdapter([], available=False, reason="no onnxruntime")
    mux = _mux(adapter=adapter, clock=clock)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)

    cmd = mux.compute(nav, _fakes.make_state(heading=7.0))
    _finite_in_range(cmd)
    assert mux.selected == "pid", "an unavailable adapter cannot be selected"
    _assert_pair(mux, "fallback", "unavailable")
    assert mux._latched is True

    # Latched, so the next tick reports 'latched' rather than 'unavailable'.
    clock.tick(0.1)
    cmd = mux.compute(nav, _fakes.make_state(heading=8.0))
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "latched")

    mux.set_controller_request("ppo")
    clock.tick(0.1)
    cmd = mux.compute(nav, _fakes.make_state(heading=9.0))
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "unavailable")


# ---------------------------------------------------------------------------- 11
def test_no_goal_autonomous_with_empty_navigator():
    """Orchestrator's row: autonomous mode, navigator has no waypoint.

    ``goal.lat``/``goal.lon`` are None, so the mux must NOT feed None into the
    policy (live NaN). It reports ``source='fallback'``, ``fault='no_goal'`` and
    emits a real PID command — not zero, not NaN.
    """
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(0.7, 0.7, 0.7))])
    navigator = _fakes.ScriptedNavigator([], index=0,
                                         state=NavigationState.WAYPOINT_NAV)
    mux = _mux(adapter=adapter, navigator=navigator, clock=clock)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.8,
                          desired_heading=45.0)

    cmd = mux.compute(nav, _fakes.make_state(heading=45.0, speed=0.4))
    _finite_in_range(cmd)
    _assert_pair(mux, "fallback", "no_goal")
    assert not all(v == 0.0 for v in (cmd.surge, cmd.sway, cmd.yaw)), \
        "no_goal must produce a real PID command, not zeros"
    assert adapter.calls == [], "the policy must not see a None goal"


def test_no_goal_does_not_run_but_three_consecutive_latch():
    """no_goal is a ladder fallback: 3 in a row latch like every other fault."""
    clock = _Clock()
    navigator = _fakes.ScriptedNavigator([], state=NavigationState.WAYPOINT_NAV)
    mux = _mux(navigator=navigator, clock=clock)
    nav = _fakes.make_nav(NavigationState.STATION_KEEP, desired_north_m=5.0,
                          desired_east_m=5.0)
    for i in range(3):
        cmd = mux.compute(nav, _fakes.make_state(heading=float(i)))
        _finite_in_range(cmd)
        clock.tick(0.1)
    _assert_pair(mux, "fallback", "latched")


# ---------------------------------------------------------------------------- 12
def test_no_emitted_step_exceeds_rate_limit_except_idle_manual():
    """Rule 6: slew every autonomous command, never the IDLE/MANUAL pass-through."""
    clock = _Clock()
    adapter = _fakes.FakeAdapter(
        [_fakes.PolicyStep(action=(1.0, -1.0, 1.0))], policy_dt_s=0.1,
        safety={"rate_limit_per_s": 2.0})
    navigator = _fakes.ScriptedNavigator(
        [Waypoint(-6.8969, 112.0521, "w1")], index=0,
        state=NavigationState.WAYPOINT_NAV)
    mux = _mux(adapter=adapter, navigator=navigator, clock=clock)
    autonomous = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.9,
                                 desired_heading=90.0)

    cmds = []
    for i in range(6):
        cmds.append(mux.compute(autonomous, _fakes.make_state(heading=float(i * 3))))
        clock.tick(0.1)
    # A fault tick in the middle of the policy run: the fallback PID command is
    # also slew-limited when a policy command was already emitted.
    cmds.append(mux.compute(autonomous, _fakes.make_state(heading=99.0,
                                                          gps_valid=False)))
    clock.tick(0.1)
    cmds.append(mux.compute(autonomous, _fakes.make_state(heading=99.5)))
    _assert_no_big_step(cmds)

    # IDLE / MANUAL take effect within one tick: a full-scale jump is allowed.
    idle_before = mux.compute(_fakes.make_nav(NavigationState.IDLE),
                              _fakes.make_state(heading=0.0))
    manual = _fakes.make_nav(NavigationState.MANUAL)
    manual_truth = ThrustComputer().compute(manual,
                                            _fakes.make_state(heading=0.0))
    manual_cmd = mux.compute(manual, _fakes.make_state(heading=0.0))
    assert (manual_cmd.surge, manual_cmd.sway, manual_cmd.yaw) == \
        (manual_truth.surge, manual_truth.sway, manual_truth.yaw), \
        "MANUAL must be a byte-exact pass-through"
    assert mux.source == "pid" and mux.fault is None
    _finite_in_range(idle_before)


def test_ladder_order_gps_outranks_imu_mag_and_geofence():
    """First match wins: gps > imu > mag, and a sensor fault outranks geofence."""
    clock = _Clock()
    navigator = _fakes.ScriptedNavigator(
        [Waypoint(-6.8969, 112.0521, "w1")], state=NavigationState.WAYPOINT_NAV)
    mux = _mux(navigator=navigator, clock=clock)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)

    far = _fakes.make_state(latitude=-6.8969 + 0.001, gps_valid=False,
                            imu_valid=False, mag_valid=False)
    mux.compute(nav, far)
    _assert_pair(mux, "fallback", "gps")

    mux = _mux(navigator=navigator, clock=_Clock())
    mux.compute(nav, _fakes.make_state(latitude=-6.8969 + 0.001,
                                       imu_valid=False, mag_valid=False))
    _assert_pair(mux, "fallback", "imu")

    mux = _mux(navigator=navigator, clock=_Clock())
    mux.compute(nav, _fakes.make_state(latitude=-6.8969 + 0.001, mag_valid=False))
    _assert_pair(mux, "fallback", "mag")
