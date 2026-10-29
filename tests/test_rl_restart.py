"""Restart safety: a fresh boot is IDLE + PID, and pre-boot commands are dropped.

Frozen contract (main.py + firebase section, and ``P1.md`` §5):

* a fresh ``ASVSystem()`` comes up with ``navigator.state is IDLE`` and
  ``controller.selected == "pid"``;
* the ``_last_command_ts`` gate set in ``FirebaseBridge.__init__`` ("ignore
  commands older than boot") drops a ``RemoteCommand`` written **before** the
  restart — including one carrying ``controller="ppo"`` — so the vessel never
  switches to PPO and never enters waypoint nav from a stale command.

These run the real ``ASVSystem`` with ``hardware_mock`` installed (see
conftest) and Firebase disabled; no network, no GPIO, no serial.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.comms.firebase_bridge import FirebaseBridge, RemoteCommand
from src.control.navigation import NavigationState


@pytest.fixture
def system():
    """A fresh ASVSystem. Firebase is disabled by default (no firebase-admin)."""
    from src.main import ASVSystem

    sys_ = ASVSystem()
    assert not sys_.firebase._enabled, (
        "test assumes FirebaseBridge is disabled; "
        "enable via FIREBASE_ENABLED and this fixture needs rework")
    return sys_


class _Event:
    """The only surface ``_on_command_event`` reads: ``.data``."""

    def __init__(self, data):
        self.data = data


def test_fresh_system_is_idle_with_pid(system):
    assert system.navigator.state is NavigationState.IDLE
    assert system.navigator.waypoints == []
    assert system.controller.selected == "pid", (
        f"fresh boot selected {system.controller.selected!r}, expected 'pid'")
    # The adapter is fail-soft and (with RL_ENABLED=False) unavailable, never a crash.
    assert system.rl_adapter is not None


def test_preboot_ppo_command_is_dropped_by_timestamp_gate(system):
    """A command stamped before boot must be discarded, controller field included."""
    bridge = system.firebase
    before_boot = bridge._last_command_ts - 60.0  # written a minute before restart

    stale = {
        "mode": "waypoint",
        "controller": "ppo",
        "timestamp": before_boot,
        "waypoints": [{"lat": -6.8960, "lon": 112.0521, "name": "stale"}],
    }

    # A queue + loop let us observe whether the command would have been dispatched.
    dispatched = []
    loop = asyncio.new_event_loop()
    try:
        bridge._cmd_loop = loop
        bridge._cmd_queue = asyncio.Queue()
        bridge._on_command_event(_Event(stale))
    finally:
        loop.close()

    assert bridge._cmd_queue.empty(), (
        "pre-boot command was dispatched — the timestamp gate did not hold")
    assert bridge._last_command_ts == pytest.approx(
        time.time(), abs=5.0), "the gate timestamp moved on a dropped command"
    # The stale PPO command must not have reached the (async) dispatch path.
    loop2 = asyncio.new_event_loop()
    try:
        dispatched = loop2.run_until_complete(_drain(bridge._cmd_queue))
    finally:
        loop2.close()
    assert dispatched == []

    # The system itself is untouched: still IDLE, still PID.
    assert system.navigator.state is NavigationState.IDLE
    assert system.controller.selected == "pid"


async def _drain(queue):
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def test_postboot_command_is_dispatched_and_parsed(system):
    """Control: a command stamped *after* boot is dispatched (proves the gate works)."""
    bridge = system.firebase
    fresh_ts = bridge._last_command_ts + 60.0
    data = {"mode": "station_keep", "controller": "ppo", "timestamp": fresh_ts}

    loop = asyncio.new_event_loop()
    try:
        bridge._cmd_loop = loop
        bridge._cmd_queue = asyncio.Queue()
        bridge._on_command_event(_Event(data))
        drained = loop.run_until_complete(_drain(bridge._cmd_queue))
    finally:
        loop.close()

    assert len(drained) == 1
    cmd = drained[0]
    assert isinstance(cmd, RemoteCommand)
    assert cmd.mode == "station_keep"
    assert cmd.controller == "ppo"
    assert bridge._last_command_ts == pytest.approx(fresh_ts)


def test_preboot_waypoint_command_never_starts_nav(system):
    """A pre-boot waypoint command must not move the navigator out of IDLE."""
    bridge = system.firebase
    stale = {
        "mode": "waypoint",
        "controller": "ppo",
        "timestamp": bridge._last_command_ts - 1.0,
        "waypoints": [{"lat": -6.8960, "lon": 112.0521, "name": "stale"}],
    }
    loop = asyncio.new_event_loop()
    try:
        bridge._cmd_loop = loop
        bridge._cmd_queue = asyncio.Queue()
        bridge._on_command_event(_Event(stale))
        dispatched = loop.run_until_complete(_drain(bridge._cmd_queue))
    finally:
        loop.close()

    assert dispatched == []
    assert system.navigator.state is NavigationState.IDLE
    assert system.navigator.current_wp_index == 0
    assert system.controller.selected == "pid"
