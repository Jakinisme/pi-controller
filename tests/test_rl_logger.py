"""T10 — ``RLLogger`` behaviour across a mixed policy/fault run.

The logger is the ground-truth trace (``src/rl/logger.py``): non-blocking
enqueue, all I/O in ``flush()``/``close()``, rotation at ``max_bytes``,
``t_monotonic`` relative to the first row. This file drives it through the real
``ControllerMux`` for a mixed policy + fault run and asserts:

* every CSV row carries the full 25-column header;
* the row count equals policy steps + source-change rows, where the asserted
  rule is: **one row per policy step, plus one row on every change of the
  mux's ``source`` value, minus the policy row that itself establishes the new
  source** (the mux suppresses the duplicate via ``_last_source_logged``);
* ``goal_lat``/``goal_lon`` are populated while the navigator has a waypoint;
* a monkeypatched slow disk never delays a control tick by >5 ms;
* rotation with ``max_bytes=200`` loses no row;
* ``t_monotonic`` is relative (first row 0.0) and the columns
  ``DRL/runtime/infer_replay.py`` *requires* — ``t, lat, lon, heading_deg,
  speed_mps`` — are all present.
"""

from __future__ import annotations

import csv
import io
import time

import pytest

from src.control.controller_mux import ControllerMux
from src.control.navigation import NavigationState, Waypoint
from src.control.thrust import ThrustComputer
from src.rl.logger import RLLogger

from tests import _fakes

EXPECTED_HEADER = (
    "t_monotonic,lat,lon,heading_deg,yaw_rate_deg_s,speed_mps,course_deg,"
    "gps_valid,imu_valid,mag_valid,goal_lat,goal_lon,mode,wp_idx,n_wp,source,"
    "fault,inference_ms,action_raw_surge,action_raw_sway,action_raw_yaw,"
    "action_applied_surge,action_applied_sway,action_applied_yaw,thrust_scale"
)
REPLAY_REQUIRED = ("t_monotonic", "lat", "lon", "heading_deg", "speed_mps")

_WPS = [Waypoint(-6.8969, 112.0521, "w1"), Waypoint(-6.8960, 112.0525, "w2")]


class _Clock:
    def __init__(self, t=3000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, dt=0.1):
        self.t += dt


def _cfg():
    return {"RL_ENABLED": True, "RL_DEFAULT_CONTROLLER": "pid",
            "RL_WATCHDOG_S": 0.5, "RL_GEOFENCE_M": 20.0,
            "RL_MAX_INFERENCE_MS": 50.0, "RL_MAX_THRUST_SCALE": 1.0}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def _data_rows(path):
    rows = _rows(path)
    assert rows, f"no rows in {path}"
    return rows[1:]


def test_mixed_run_full_header_and_row_count_rule(tmp_path):
    """Policy steps + one row per non-policy source entry, every row full-width.

    Asserted rule (read off ``mux._emit_log_row`` / ``_log_source_change``):

        rows == policy_steps + fallback_entries

    where ``policy_steps`` is the number of ``adapter.step`` calls (each writes
    exactly one row via ``_emit_log_row``, which also sets
    ``_last_source_logged = "policy"``), and ``fallback_entries`` counts one row
    each time the mux enters a non-policy source from a *different* logged
    source — a return to ``policy`` writes no extra row because the next policy
    step's own row carries ``source="policy"``.
    """
    clock = _Clock()
    adapter = _fakes.FakeAdapter(
        [_fakes.PolicyStep(action=(0.3, 0.1, 0.0)),
         _fakes.PolicyStep(action=(0.4, 0.2, 0.0))],
        policy_dt_s=0.1, safety={"rate_limit_per_s": 2.0})
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.6,
                          desired_heading=45.0)
    with RLLogger(tmp_path, flush_interval_s=0.0, clock=clock) as logger:
        mux = ControllerMux(
            ThrustComputer(), adapter,
            _fakes.ScriptedNavigator(_WPS, index=0,
                                     state=NavigationState.WAYPOINT_NAV),
            gps=_fakes.FakeGPS(true_course=15.0), config=_cfg(), clock=clock,
            rl_logger=logger)
        mux.set_controller_request("ppo")

        # 4 policy ticks (0.1 s apart), then a gps fault, then 2 more good ticks.
        script = [None, None, None, None, "fault", None, None]
        observed_sources = []
        for i, kind in enumerate(script):
            state = _fakes.make_state(heading=float(i),
                                      gps_valid=(kind != "fault"))
            mux.compute(nav, state)
            observed_sources.append(mux.source)
            clock.tick(0.1)
        policy_steps = len(adapter.calls)

        logger.flush()   # flush_interval_s=0.0 -> writes now
        path = logger.path
        assert path is not None and path.is_file()

    rows = _rows(path)
    assert rows[0] == EXPECTED_HEADER.split(","), "header must be the frozen 25"
    assert len(rows[0]) == 25

    for i, row in enumerate(rows[1:], start=2):
        assert len(row) == 25, f"row {i} has {len(row)} columns, want 25: {row!r}"

    # One fallback entry in the script above.
    fallback_entries = sum(1 for s in observed_sources if s != "policy")
    assert fallback_entries == 1, f"script drift: {observed_sources}"
    assert len(rows) - 1 == policy_steps + fallback_entries, (
        f"row count {len(rows) - 1} != policy_steps {policy_steps} "
        f"+ fallback_entries {fallback_entries} (sources: {observed_sources})")

    sources = [r[15] for r in rows[1:]]
    assert sources == ["policy", "policy", "policy", "policy", "fallback",
                       "policy", "policy"], sources
    assert [r[16] for r in rows[1:]][4] == "gps"


def test_row_count_rule_is_policy_steps_plus_source_changes(tmp_path):
    """Pin the exact rule with a hand-built, fully deterministic run.

    3 policy steps then a permanent gps fault for 3 ticks:
      rows = 3 policy steps + [policy transition] + [fallback,gps transition]
             - 1 (the first policy row already established source='policy')
           = 3 + 2 - 1 = 4.
    The second and third fault ticks are neither policy steps nor source
    changes, so they log nothing.
    """
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(0.2, 0.0, 0.0))],
                                 policy_dt_s=0.1, safety={"rate_limit_per_s": 2.0})
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)
    with RLLogger(tmp_path, flush_interval_s=0.0, clock=clock) as logger:
        mux = ControllerMux(
            ThrustComputer(), adapter,
            _fakes.ScriptedNavigator(_WPS, state=NavigationState.WAYPOINT_NAV),
            gps=_fakes.FakeGPS(), config=_cfg(), clock=clock, rl_logger=logger)
        mux.set_controller_request("ppo")
        for i in range(3):                     # 3 policy steps (0.1 s apart)
            mux.compute(nav, _fakes.make_state(heading=float(i)))
            clock.tick(0.1)
        for i in range(3):                     # 3 consecutive gps-fault ticks
            mux.compute(nav, _fakes.make_state(heading=float(50 + i), gps_valid=False))
            clock.tick(0.1)
        logger.flush()
        path = logger.path

    step_calls = 3
    rows = _rows(path)
    assert len(rows) - 1 == step_calls + 2 - 1, \
        f"expected 4 rows (3 policy + policy transition + gps transition - 1), " \
        f"got {len(rows) - 1}: {rows[1:]}"
    assert [r[15] for r in rows[1:]] == ["policy", "policy", "policy", "fallback"]
    assert rows[-1][16] == "gps"


def test_goal_lat_lon_populated_with_waypoint(tmp_path):
    """With a current waypoint, ``goal_lat``/``goal_lon`` are the waypoint's."""
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(0.2, 0.0, 0.0))],
                                 policy_dt_s=0.1, safety={"rate_limit_per_s": 2.0})
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)
    with RLLogger(tmp_path, flush_interval_s=0.0, clock=clock) as logger:
        mux = ControllerMux(
            ThrustComputer(), adapter,
            _fakes.ScriptedNavigator(_WPS, index=0,
                                     state=NavigationState.WAYPOINT_NAV),
            gps=_fakes.FakeGPS(), config=_cfg(), clock=clock, rl_logger=logger)
        mux.set_controller_request("ppo")
        mux.compute(nav, _fakes.make_state(heading=3.0))   # step 1, wp 0
        clock.tick(0.1)
        # Advance to wp 1: a fresh goal, logged on the next policy step.
        mux.navigator.current_wp_index = 1
        mux.compute(nav, _fakes.make_state(heading=4.0))
        clock.tick(0.1)
        logger.flush()
        path = logger.path

    rows = _data_rows(path)
    assert rows, "no data rows written"
    hdr = EXPECTED_HEADER.split(",")
    gi, gj = hdr.index("goal_lat"), hdr.index("goal_lon")
    for row in rows:
        assert row[gi], f"goal_lat empty on a waypoint run: {row!r}"
        assert row[gj], f"goal_lon empty on a waypoint run: {row!r}"
    got = {(round(float(r[gi]), 4), round(float(r[gj]), 4)) for r in rows}
    assert got == {(round(_WPS[0].latitude, 4), round(_WPS[0].longitude, 4)),
                   (round(_WPS[1].latitude, 4), round(_WPS[1].longitude, 4))}, got


def test_goal_lat_lon_blank_without_waypoint(tmp_path):
    """No current waypoint -> the goal columns are honestly blank, not 0."""
    clock = _Clock()
    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(0.2, 0.0, 0.0))],
                                 policy_dt_s=0.1, safety={"rate_limit_per_s": 2.0})
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)
    with RLLogger(tmp_path, flush_interval_s=0.0, clock=clock) as logger:
        mux = ControllerMux(
            ThrustComputer(), adapter,
            _fakes.ScriptedNavigator([], state=NavigationState.WAYPOINT_NAV),
            gps=_fakes.FakeGPS(), config=_cfg(), clock=clock, rl_logger=logger)
        mux.set_controller_request("ppo")
        mux.compute(nav, _fakes.make_state(heading=3.0))   # no_goal -> fallback row
        clock.tick(0.1)
        logger.flush()
        path = logger.path

    rows = _data_rows(path)
    hdr = EXPECTED_HEADER.split(",")
    gi, gj = hdr.index("goal_lat"), hdr.index("goal_lon")
    for row in rows:
        assert row[gi] == "" and row[gj] == "", f"goal must be blank: {row!r}"


def test_slow_disk_never_delays_a_control_tick(tmp_path, monkeypatch):
    """A 50 ms/row write path must not delay ``logger.log()`` or a mux tick.

    ``log()`` is a pure enqueue, so the slow path lives in ``flush()``/``close()``.
    The assertion bounds the *tick* cost: the time spent inside ``log()`` and
    inside ``mux.compute`` for the whole run stays far below 5 ms.
    """
    clock = _Clock()
    slow_writes = {"n": 0}

    real_format = RLLogger._format

    def slow_format(self, row):
        slow_writes["n"] += 1
        time.sleep(0.05)          # 50 ms "disk"
        return real_format(self, row)

    # The slow path is in the write/format stage, reached only from flush().
    monkeypatch.setattr(RLLogger, "_format", slow_format)

    adapter = _fakes.FakeAdapter([_fakes.PolicyStep(action=(0.2, 0.0, 0.0))],
                                 policy_dt_s=0.1, safety={"rate_limit_per_s": 2.0})
    nav = _fakes.make_nav(NavigationState.WAYPOINT_NAV, desired_speed=0.5)
    logger = RLLogger(tmp_path, flush_interval_s=10.0, clock=clock)
    mux = ControllerMux(
        ThrustComputer(), adapter,
        _fakes.ScriptedNavigator(_WPS, state=NavigationState.WAYPOINT_NAV),
        gps=_fakes.FakeGPS(), config=_cfg(), clock=clock, rl_logger=logger)
    mux.set_controller_request("ppo")

    log_ns = 0
    tick_ns = 0
    for i in range(20):
        t0 = time.perf_counter_ns()
        mux.compute(nav, _fakes.make_state(heading=float(i)))
        tick_ns += time.perf_counter_ns() - t0
        clock.tick(0.1)

    # Nothing has hit disk yet (flush_interval_s=10 and the clock is fake/idle).
    assert slow_writes["n"] == 0, "log() must not touch the slow disk path"
    assert (tick_ns / 1_000_000) < 5.0, \
        f"20 ticks cost {tick_ns / 1e6:.2f} ms inside compute (limit 5 ms per tick)"

    t1 = time.perf_counter_ns()
    logger.log({"t_monotonic": clock()})
    log_ns = time.perf_counter_ns() - t1
    assert log_ns / 1_000_000 < 5.0, f"log() took {log_ns / 1e6:.3f} ms (limit 5 ms)"

    # close() pays the slow disk cost; every row still lands.
    logger.close()
    assert slow_writes["n"] >= 1, "the slow path must actually have been exercised"
    parts = sorted(tmp_path.glob("rl_*.csv"))
    assert parts, f"no CSV written into {tmp_path}"
    rows = _data_rows(parts[0])
    assert len(rows) == 20 + 1, f"expected 21 rows after close, got {len(rows)}"


def test_rotation_with_small_max_bytes_loses_no_row(tmp_path):
    """``max_bytes=200`` rotates to ``_part2``/``_part3`` ... with no lost row."""
    clock = _Clock()
    with RLLogger(tmp_path, max_bytes=200, flush_interval_s=0.0, clock=clock) as logger:
        for i in range(6):
            logger.log({"t_monotonic": float(i), "lat": -6.8969, "lon": 112.0521,
                        "heading_deg": float(i), "speed_mps": 0.5, "source": "policy",
                        "fault": None, "mode": "waypoint"})
            clock.tick(0.1)
        logger.close()

    parts = sorted(p.name for p in tmp_path.glob("rl_*.csv"))
    assert len(parts) >= 3, f"expected rotation into >=3 parts, got {parts!r}"
    assert any("_part2.csv" in p for p in parts), parts
    assert any("_part3.csv" in p for p in parts), parts

    total = 0
    for p in parts:
        rows = _rows(tmp_path / p)
        assert rows[0] == EXPECTED_HEADER.split(","), f"{p}: header must be 25 cols"
        for r in rows[1:]:
            assert len(r) == 25, f"{p}: short row {r!r}"
        total += len(rows) - 1
    assert total == 6, f"rotation lost a row: {total} data rows across {parts!r}"


def test_t_monotonic_relative_and_replay_columns_present(tmp_path):
    """First row is 0.0 and the replay-required columns all exist."""
    clock = _Clock(t=12345.0)   # epoch-scale fake monotonic
    with RLLogger(tmp_path, flush_interval_s=0.0, clock=clock) as logger:
        for i in range(3):
            logger.log({"t_monotonic": 12345.0 + 0.1 * i})
            clock.tick(0.1)
        logger.flush()
        path = logger.path

    rows = _rows(path)
    hdr = rows[0]
    for col in REPLAY_REQUIRED:
        assert col in hdr, f"replay-required column {col!r} missing from the header"
    assert hdr == EXPECTED_HEADER.split(",")

    first = float(rows[1][hdr.index("t_monotonic")])
    assert first == 0.0, f"first t_monotonic must be 0.0, got {first!r}"
    for i, row in enumerate(rows[1:], start=0):
        t = float(row[hdr.index("t_monotonic")])
        assert abs(t - 0.1 * i) < 1e-6, f"row {i}: t={t!r}, want {0.1 * i}"
        assert t < 1.0, f"t_monotonic must be RELATIVE, got {t!r} (epoch-scale)"


def test_log_never_raises_and_drops_unusable_rows(tmp_path):
    """``log()`` never raises; rows it cannot copy or format are dropped, not fatal.

    ``log()`` does ``deque.append(dict(row))`` inside a ``try/except: pass``, so
    the two ``log(None)`` calls are silently dropped. Downstream, ``_write_all``
    drops any row ``_format`` cannot render (the non-numeric ``t_monotonic`` and
    the NaN) without taking the rest of the queue with it, so the one clean row
    still lands. Dropping a NaN row is deliberate: a NaN in the CSV would corrupt
    every consumer of the sysid trace.
    """
    clock = _Clock()
    logger = RLLogger(tmp_path, flush_interval_s=0.0, clock=clock)
    logger.log(None)                       # dict(None) raises -> dropped
    logger.log({"t_monotonic": "not-a-number"})   # unformattable -> dropped
    logger.log({"heading_deg": float("nan")})     # NaN -> dropped on purpose
    logger.log(None)                       # dropped
    logger.log({"lat": 1.0, "lon": 2.0, "heading_deg": 3.0, "speed_mps": 0.1})
    logger.close()
    parts = sorted(tmp_path.glob("rl_*.csv"))
    assert parts, f"no CSV written into {tmp_path}"
    rows = _data_rows(parts[0])
    assert len(rows) == 1, f"expected 1 formattable row, got {len(rows)}"
    assert all(len(r) == 25 for r in rows)


def test_unformattable_row_is_dropped_and_the_rest_land(tmp_path):
    """A row the logger cannot format is dropped; rows queued behind it still land.

    This is the fixed behaviour (see reports/P1_T10_tests.md): ``_write_all``
    catches a ``_format`` failure per row, counts it in ``_errors`` and keeps
    going, instead of aborting the flush and silently losing the trace tail.
    """
    clock = _Clock()
    logger = RLLogger(tmp_path, flush_interval_s=0.0, clock=clock)
    logger.log({"t_monotonic": "not-a-number"})
    logger.log({"t_monotonic": 1.0, "lat": 5.0, "lon": 6.0})
    logger.close()

    assert logger._errors == 1, "the write error is counted, not raised"
    parts = sorted(tmp_path.glob("rl_*.csv"))
    rows = _data_rows(parts[0])
    assert len(rows) == 1, "only the formattable row is written"
    assert rows[0][0] == "0.000000", "the surviving row is the counter-based one"
