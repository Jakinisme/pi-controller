"""Regression: the estimator timestamp must live in RLController's clock domain.

``RLController._check`` compares ``estimator_state["t"]`` against **its own**
clock (``time.monotonic`` by default, injectable). The mux used to stamp that
field with its own ``now`` argument, so any caller driving the mux with a
different clock — an injected clock, a replay driver, ``runtime/infer_replay.py``
— looked like an ancient GPS fix and the policy fell back to zero thrust
**forever** with ``fault="watchdog"``.

These tests use a real ``RLAdapter`` over the real R7 ONNX so the clock wiring is
exercised end to end, not through a fake. Skipped cleanly when onnxruntime or the
model is absent.
"""

from __future__ import annotations

import os

import pytest

onnxruntime = pytest.importorskip("onnxruntime")

from tests import _fakes  # noqa: E402
from tests.conftest import META_PATH, MODEL_PATH  # noqa: E402
from src.control.navigation import NavigationState  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(MODEL_PATH) and os.path.isfile(META_PATH)),
    reason=f"R7 model/meta absent ({MODEL_PATH})",
)


class _VirtualClock:
    """A clock in its own domain, starting at 0.0 like a replay driver would."""

    def __init__(self, start: float = 0.0, step: float = 0.1):
        self.t = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        return self.t

    def tick(self) -> None:
        self.t += self.step


def _mux(clock):
    """A ControllerMux on a real RLAdapter, forced to the policy, with ``clock``."""
    from src.control.controller_mux import ControllerMux
    from src.control.navigation import NavigationState, Waypoint
    from src.control.thrust import ThrustComputer
    from src.rl.adapter import RLAdapter

    adapter = RLAdapter(model_path=MODEL_PATH, meta_path=META_PATH, clock=clock)
    if not adapter.available:
        pytest.skip(f"adapter unavailable: {adapter.reason}")
    nav = _fakes.ScriptedNavigator(
        waypoints=[Waypoint(latitude=-6.8960, longitude=112.0521, name="N")],
        state=NavigationState.WAYPOINT_NAV)
    return ControllerMux(ThrustComputer(), adapter, nav, None,
                         config={"RL_ENABLED": True, "RL_DEFAULT_CONTROLLER": "pid"})


def test_virtual_clock_does_not_trip_the_watchdog():
    """A clock starting at 0.0 must still yield ``source == "policy"``.

    This is the exact bug: the mux's own ``now`` starts near 0 in a replay or a
    test, RLController's clock is real monotonic (~1e5 s), so ``now - t`` was
    hugely negative the other way round and every tick read ``watchdog``.
    """
    clock = _VirtualClock()
    mux = _mux(clock)
    mux.set_controller_request("ppo")

    sources = []
    for _ in range(5):
        clock.tick()
        mux.compute(_fakes.make_nav(NavigationState.WAYPOINT_NAV,
                                    desired_speed=0.5, desired_heading=0.0),
                    _fakes.make_state(), now=clock())
        sources.append(mux.source)

    assert "policy" in sources, f"never reached the policy; sources={sources}"
    assert "watchdog" not in [str(f) for f in (mux.fault,)], (
        f"watchdog latched under a virtual clock: source={mux.source} fault={mux.fault}")


def test_est_timestamp_comes_from_the_adapter_clock():
    """Direct check of the contract the fix restores."""
    clock = _VirtualClock(start=1234.5)
    mux = _mux(clock)
    est = mux._build_est(_fakes.make_state(), now=999.0)
    assert est["t"] == pytest.approx(1234.5), (
        f"est['t'] must be adapter-clock time 1234.5, got {est['t']} "
        "(999.0 means the mux `now` leaked through)")
