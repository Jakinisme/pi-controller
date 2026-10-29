"""Shared test doubles for the RL integration tests (T09).

Everything the test files need lives here once: a scripted adapter, a scripted
navigator, a fused-state builder and a scripted navigation-output builder. No
test file may re-implement any of it.

Signatures mirror the frozen contracts:
  * FakeAdapter   -> src.rl.adapter.RLAdapter surface the mux actually touches
                     (``available``, ``step``, ``reset``, ``policy_dt_s``,
                     ``last_inference_ms``, ``meta``, ``safety``)
  * ScriptedNavigator -> enough of WaypointNavigator for the mux's calls
                     (``stop()``, ``current_waypoint``, ``waypoints``,
                     ``current_wp_index``)
  * make_state(nav)   -> src.sensors.fusion.FusedState (real class, never a copy)
  * make_nav(...)     -> src.control.navigation.NavigationOutput
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from src.control.navigation import NavigationOutput, NavigationState, Waypoint
from src.sensors.fusion import FusedState

__all__ = [
    "PolicyStep",
    "FakeAdapter",
    "ScriptedNavigator",
    "make_state",
    "make_nav",
    "FakeGPS",
]


@dataclass
class PolicyStep:
    """One scripted adapter result."""

    action: tuple = (0.0, 0.0, 0.0)
    source: str = "policy"
    fault: Optional[str] = None
    inference_ms: float = 0.0
    raises: Optional[BaseException] = None
    raw_action: Optional[tuple] = None


class FakeAdapter:
    """Scripted stand-in for :class:`src.rl.adapter.RLAdapter`.

    ``results`` is consumed one entry per ``step()`` call; the last entry
    repeats when the script runs dry (so a long tick loop needs no long list).
    A ``PolicyStep`` with ``raises`` set raises it instead of returning.
    """

    def __init__(self, results: Sequence[PolicyStep] = (), *, available: bool = True,
                 reason: Optional[str] = None, policy_dt_s: float = 0.1,
                 meta: Optional[dict] = None, safety: Optional[dict] = None):
        self.results: List[PolicyStep] = list(results)
        self.available = bool(available)
        self.reason = reason
        self.policy_dt_s = float(policy_dt_s)
        self.meta = dict(meta or {"policy_dt_s": self.policy_dt_s})
        self.safety = dict(safety or {"rate_limit_per_s": 2.0})
        self.last_inference_ms = 0.0
        self.calls: List[tuple] = []
        self.resets = 0

    # ------------------------------------------------------------- interface
    def step(self, est, goal):
        self.calls.append((dict(est), dict(goal)))
        if not self.available:
            raise RuntimeError(f"adapter unavailable: {self.reason}")
        if not self.results:
            raise AssertionError("FakeAdapter.step called with an empty script")
        idx = min(len(self.calls) - 1, len(self.results) - 1)
        result = self.results[idx]
        if result.raises is not None:
            raise result.raises
        self.last_inference_ms = result.inference_ms
        action = tuple(float(a) for a in result.action)
        raw = tuple(action if result.raw_action is None else result.raw_action)
        return _FakePolicyResult(action=action, raw_action=raw,
                                 source=result.source, fault=result.fault,
                                 inference_ms=result.inference_ms)

    def reset(self) -> None:
        self.resets += 1
        self.calls.clear()
        self.last_inference_ms = 0.0

    @property
    def status(self) -> dict:
        return {"source": "init", "fault": None, "action": (0.0, 0.0, 0.0)}


@dataclass(frozen=True)
class _FakePolicyResult:
    """Duck-typed PolicyResult (the real class lives in src.rl.adapter)."""

    action: tuple
    raw_action: tuple
    source: str
    fault: Optional[str]
    inference_ms: float


class ScriptedNavigator:
    """Minimal navigator: a fixed waypoint list + index + a recorded stop()."""

    def __init__(self, waypoints: Optional[Sequence[Waypoint]] = None,
                 index: int = 0, state: NavigationState = NavigationState.IDLE):
        self.waypoints = list(waypoints or [])
        self.current_wp_index = index
        self.state = state
        self.stopped = 0
        self.manual: Optional[tuple] = None

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_wp_index < len(self.waypoints):
            return self.waypoints[self.current_wp_index]
        return None

    def stop(self) -> None:
        self.stopped += 1
        self.state = NavigationState.IDLE

    def start_waypoint_nav(self, waypoints) -> None:
        self.waypoints = list(waypoints)
        self.current_wp_index = 0
        self.state = NavigationState.WAYPOINT_NAV


def make_state(**kw) -> FusedState:
    """A FusedState valid for the policy by default; override any field."""
    base = dict(
        latitude=-6.8969, longitude=112.0521, heading=0.0, speed=0.0,
        roll=0.0, pitch=0.0, yaw_rate=0.0, timestamp=time.time(),
        gps_valid=True, imu_valid=True, mag_valid=True,
    )
    base.update(kw)
    return FusedState(**base)


def make_nav(mode: NavigationState = NavigationState.IDLE, **kw) -> NavigationOutput:
    """A NavigationOutput in the requested mode; override any field."""
    base = dict(
        desired_speed=0.0, desired_heading=0.0, desired_north_m=0.0,
        desired_east_m=0.0, hold_heading=0.0, mode=mode,
    )
    base.update(kw)
    return NavigationOutput(**base)


class FakeGPS:
    """The only GPS surface the mux reads: ``data.true_course``."""

    def __init__(self, true_course: float = 0.0):
        self.data = type("_D", (), {"true_course": true_course})()
