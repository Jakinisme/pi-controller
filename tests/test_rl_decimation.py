"""T10 — decimation and jitter behaviour of ``ControllerMux``.

Frozen rule 3 (``p1_interfaces.md``): when ``selected == "ppo"`` the mux runs
``adapter.step`` at ``policy_dt_s`` (default 0.1 s) instead of every tick. The
condition is ``now - last_step_t >= policy_dt_s * 0.9``; the first tick always
steps. In-between ticks return a ``ThrustCommand`` carrying the *last applied*
policy action verbatim — no inference.

Jitter: when the measured gap deviates more than 30 % from ``policy_dt_s`` the
mux emits one ``log.warning`` whose message carries the gap in ms, rate-limited
to one per 5 s.

All clocks here are fakes passed to the mux (see the module note in
``test_rl_faults.py``: the adapter shares the clock in production).
"""

from __future__ import annotations

import logging

from src.control.controller_mux import ControllerMux
from src.control.navigation import NavigationState, Waypoint
from src.control.thrust import ThrustComputer

from tests import _fakes

DT = 0.1          # policy_dt_s
TICK = 0.05       # 20 Hz control loop
N_TICKS = 20      # 1.0 s of ticks -> 10 policy steps
_WPS = [Waypoint(-6.8969, 112.0521, "w1"), Waypoint(-6.8960, 112.0525, "w2")]


class _Clock:
    def __init__(self, t=2000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, dt=TICK):
        self.t += dt


def _scripted_adapter(n_actions=40):
    """Distinct action per call so a fresh step is distinguishable from a repeat."""
    steps = [_fakes.PolicyStep(action=(0.05 * i, 0.0, 0.0)) for i in range(1, n_actions + 1)]
    return _fakes.FakeAdapter(steps, policy_dt_s=DT, safety={"rate_limit_per_s": 2.0})


def _mux(clock, adapter=None, logger=None, cfg=None):
    adapter = adapter or _scripted_adapter()
    nav = _fakes.ScriptedNavigator(_WPS, index=0, state=NavigationState.WAYPOINT_NAV)
    mux = ControllerMux(ThrustComputer(), adapter, nav,
                        gps=_fakes.FakeGPS(true_course=15.0),
                        config=cfg or {"RL_ENABLED": True,
                                       "RL_DEFAULT_CONTROLLER": "pid",
                                       "RL_WATCHDOG_S": 0.5,
                                       "RL_GEOFENCE_M": 20.0,
                                       "RL_MAX_INFERENCE_MS": 50.0,
                                       "RL_MAX_THRUST_SCALE": 1.0},
                        clock=clock, logger=logger)
    mux.set_controller_request("ppo")
    return mux


def test_twenty_ticks_fifty_ms_yield_exactly_ten_steps():
    """20 ticks at 50 ms, policy_dt 0.1 s -> exactly 10 ``adapter.step`` calls."""
    clock = _Clock()
    adapter = _scripted_adapter()
    mux = _mux(clock, adapter)

    cmds = []
    for i in range(N_TICKS):
        # A fresh heading each tick so the mux's value-keyed watchdog never fires.
        cmds.append(mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV,
                                                desired_speed=0.6,
                                                desired_heading=45.0),
                                _fakes.make_state(heading=float(i))))
        clock.tick(TICK)

    assert len(adapter.calls) == 10, \
        f"expected exactly 10 policy calls in {N_TICKS} ticks, got {len(adapter.calls)}"
    assert mux.selected == "ppo"
    assert mux.source == "policy"


def test_in_between_ticks_repeat_previous_command_verbatim():
    """Ticks 1,3,5,... (odd, even 0) repeat the last applied command exactly."""
    clock = _Clock()
    adapter = _scripted_adapter()
    mux = _mux(clock, adapter)

    cmds = []
    for i in range(N_TICKS):
        cmds.append(mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV,
                                                desired_speed=0.6,
                                                desired_heading=45.0),
                                _fakes.make_state(heading=float(i))))
        clock.tick(TICK)

    steps = [i for i in range(N_TICKS) if i % 2 == 0]
    repeats = [i for i in range(N_TICKS) if i % 2 == 1]
    assert len(steps) == 10 and len(repeats) == 10

    for i in repeats:
        prev = cmds[i - 1]
        cur = cmds[i]
        assert (cur.surge, cur.sway, cur.yaw) == (prev.surge, prev.sway, prev.yaw), \
            f"tick {i} must repeat tick {i - 1} verbatim, got " \
            f"{(cur.surge, cur.sway, cur.yaw)} vs {(prev.surge, prev.sway, prev.yaw)}"

    # A repeat is a genuine repeat: repeated pairs are identical, step-to-step
    # pairs are not (the scripted actions differ by 0.05 each call).
    assert (cmds[0].surge, cmds[1].surge) != (cmds[2].surge, cmds[3].surge), \
        "the scripted steps must differ so a repeat is distinguishable"


def test_jitter_warning_fires_over_thirty_percent(caplog):
    """A 2x gap (100 % off) logs exactly one jitter warning naming the gap in ms."""
    clock = _Clock()
    adapter = _scripted_adapter()
    mux = _mux(clock, adapter)

    with caplog.at_level(logging.WARNING, logger="controller_mux"):
        mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5),
                    _fakes.make_state(heading=1.0))      # step 1 at t0
        clock.tick(0.1)
        mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5),
                    _fakes.make_state(heading=2.0))      # step 2, gap 0.1 -> on time
        clock.tick(0.2)                              # gap 0.2 = 100 % off
        mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5),
                    _fakes.make_state(heading=3.0))      # step 3 -> jitter warning

    warns = [r.message for r in caplog.records
             if r.levelno == logging.WARNING and "jitter" in r.message.lower()]
    assert len(warns) == 1, f"expected exactly 1 jitter warning, got {warns!r}"
    assert "200.0 ms" in warns[0], f"warning must carry the gap in ms: {warns[0]!r}"
    assert len(adapter.calls) == 3


def test_no_jitter_warning_within_thirty_percent(caplog):
    """A gap inside ±30 % must not warn."""
    clock = _Clock()
    adapter = _scripted_adapter()
    mux = _mux(clock, adapter)

    with caplog.at_level(logging.WARNING, logger="controller_mux"):
        mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5),
                    _fakes.make_state(heading=1.0))
        clock.tick(0.1)   # exactly on target
        mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5),
                    _fakes.make_state(heading=2.0))
        clock.tick(0.11)  # 10 % off, inside the +-30 % band
        mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5),
                    _fakes.make_state(heading=3.0))

    warns = [r.message for r in caplog.records
             if r.levelno == logging.WARNING and "jitter" in r.message.lower()]
    assert warns == [], f"no warning expected inside the band, got {warns!r}"
    assert len(adapter.calls) == 3


def test_jitter_warning_rate_limited_over_a_burst(caplog):
    """A burst of 20 x 200 ms gaps (>5 s of fake time) warns at most 3 times.

    The rate limit is one warning per 5 s of the injected clock; 20 steps at
    200 ms each is 4.0 s of elapsed time, so only warnings at t0, one ≥5 s later
    fit. Concretely: with a warn interval of 5 s and 4 s of ticks, exactly one
    warning can fire (plus one for the first over-threshold gap).

    The burst is driven purely through ``_check_jitter``'s call site
    (``_run_policy``), i.e. real ``compute`` calls, no private poke.
    """
    clock = _Clock()
    adapter = _scripted_adapter(n_actions=60)
    mux = _mux(clock, adapter)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)

    with caplog.at_level(logging.WARNING, logger="controller_mux"):
        for i in range(20):
            clock.tick(0.2)     # 200 ms: 100 % off policy_dt_s
            mux.compute(nav, _fakes.make_state(heading=float(i)))
        # Every tick after the first is over threshold: 19 candidate warnings,
        # 4.0 s of fake time, 5 s limit -> exactly 1 survives.

    warns = [r.message for r in caplog.records
             if r.levelno == logging.WARNING and "jitter" in r.message.lower()]
    assert len(adapter.calls) == 20, "every 200 ms tick must step"
    assert len(warns) == 1, (
        f"rate limit is 1 per 5 s; 4.0 s of 200 ms ticks must warn once, "
        f"got {len(warns)}: {warns!r}")

    # Cross the 5 s boundary: one more warning is allowed.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="controller_mux"):
        for i in range(40):
            clock.tick(0.2)
            mux.compute(nav, _fakes.make_state(heading=float(100 + i)))
    warns2 = [r.message for r in caplog.records
              if r.levelno == logging.WARNING and "jitter" in r.message.lower()]
    assert 1 <= len(warns2) <= 3, (
        f"8 s of 200 ms ticks must warn 1-2 times, got {len(warns2)}: {warns2!r}")


def test_first_tick_always_steps_regardless_of_jitter():
    """``last_step_t is None`` -> the first tick steps (rule 3)."""
    clock = _Clock()
    adapter = _scripted_adapter()
    mux = _mux(clock, adapter)
    mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5),
                _fakes.make_state(heading=1.0))
    assert len(adapter.calls) == 1
    assert adapter.calls[0][1].get("lat") == _WPS[0].latitude
    assert adapter.calls[0][1].get("lon") == _WPS[0].longitude


def test_repeat_does_not_call_step_and_keeps_finite_range():
    """In-between ticks never step and stay inside [-1, 1]."""
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(1.0, -1.0, 0.5))],
                                 policy_dt_s=DT, safety={"rate_limit_per_s": 2.0})
    mux = _mux(clock, adapter)
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)

    mux.compute(nav, _fakes.make_state(heading=1.0))     # step
    clock.tick(TICK)                                     # 0.05 < 0.09 -> repeat
    cmd = mux.compute(nav, _fakes.make_state(heading=2.0))
    assert len(adapter.calls) == 1, "a 50 ms tick must not re-run the policy"
    assert all(-1.0 <= v <= 1.0 for v in (cmd.surge, cmd.sway, cmd.yaw))
