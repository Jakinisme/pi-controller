"""pytest bootstrap for the RL integration tests (T09).

Mirrors ``test.py``'s header exactly: the repo root goes on ``sys.path`` and
``hardware_mock.install()`` runs **before any** ``src.*`` import, so importing
``src.main`` / ``src.sensors.*`` needs no GPIO, no pigpio, no smbus2, no serial.
Do not reorder those two lines.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hardware_mock import install  # noqa: E402  (must precede any src.* import)

install()

from tests import _fakes  # noqa: E402
from src.sensors.fusion import FusedState  # noqa: E402

#: DRL lives here on this box; both fixtures below skip cleanly when absent.
DRL_PATH = os.environ.get("DRL_PATH", r"E:/buoy-project/drl")

MODEL_PATH = os.path.join(REPO_ROOT, "models", "policy.onnx")
META_PATH = os.path.join(REPO_ROOT, "models", "policy_meta.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "models", "MANIFEST.json")


# --------------------------------------------------------------- repo fixtures
@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def drl_path() -> str:
    """The read-only DRL repo, or a skip — never a failure."""
    if not os.path.isdir(DRL_PATH):
        pytest.skip(f"DRL repo absent: {DRL_PATH} (set DRL_PATH to override)")
    if not os.path.isdir(os.path.join(DRL_PATH, "runtime")):
        pytest.skip(f"DRL repo at {DRL_PATH} has no runtime/ package")
    return DRL_PATH


# ------------------------------------------------------------ state/nav doubles
@pytest.fixture
def state() -> FusedState:
    """A policy-valid fused state."""
    return _fakes.make_state()


@pytest.fixture
def state_factory():
    """``state_factory(gps_valid=False, ...)`` -> FusedState."""
    return _fakes.make_state


@pytest.fixture
def nav_factory():
    """``nav_factory(NavigationState.WAYPOINT_NAV, desired_speed=0.5)``."""
    return _fakes.make_nav


@pytest.fixture
def fake_adapter():
    """Factory: ``fake_adapter(PolicyStep((1,0,0)))`` -> FakeAdapter."""
    return _fakes.FakeAdapter


@pytest.fixture
def scripted_navigator():
    """Factory: ``scripted_navigator([Waypoint(...)])`` -> ScriptedNavigator."""
    return _fakes.ScriptedNavigator


@pytest.fixture
def fake_gps():
    """Factory: ``fake_gps(true_course=180.0)`` -> FakeGPS."""
    return _fakes.FakeGPS


# ------------------------------------------------------------ real-RL fixtures
@pytest.fixture(scope="session")
def real_model_available() -> bool:
    missing = [p for p in (MODEL_PATH, META_PATH) if not os.path.isfile(p)]
    if missing:
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture
def real_adapter(real_model_available):
    """A real :class:`RLAdapter` against ``models/policy.onnx``.

    Skips (with a clear reason) when the model/meta or onnxruntime is missing.
    """
    if not real_model_available:
        missing = [p for p in (MODEL_PATH, META_PATH) if not os.path.isfile(p)]
        reason = (f"missing model files: {missing}" if missing
                  else "onnxruntime not importable")
        pytest.skip(f"real RLAdapter unavailable: {reason}")
    from src.rl.adapter import RLAdapter

    adapter = RLAdapter(model_path=MODEL_PATH, meta_path=META_PATH)
    if not adapter.available:
        pytest.skip(f"real RLAdapter not available: {adapter.reason}")
    return adapter
