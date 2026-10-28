"""Import safety: the RL layer must never take the system down.

Two failure modes are injected in **separate subprocesses** (a poisoned
``sys.modules`` / ``sys.meta_path`` blocker cannot be undone once it leaks into
the pytest process, and ``src.main`` caches its imports):

a. ``onnxruntime`` unimportable — blocked with a ``sys.meta_path`` finder;
b. the model file missing (``RL_MODEL_PATH`` pointed at nowhere).

In both, this must hold:

* ``RLAdapter`` reports ``available=False`` **with a reason**, never raising;
* ``import src.main`` still succeeds;
* the PID path still runs a control tick (a real ``ControllerMux.compute`` with
  RL off produces a ``ThrustCommand`` and the policy never runs).

The child scripts import ``hardware_mock`` and call ``install()`` before any
``src.*`` import, exactly like the test suite's own header.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT, MODEL_PATH

PYTHON = sys.executable

_PREAMBLE = """
import os, sys
sys.path.insert(0, {root!r})
from hardware_mock import install
install()
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, "-c", script], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=180)


def test_onnxruntime_hidden_in_subprocess():
    """onnxruntime blocked at sys.meta_path: adapter fail-soft, main imports, PID ticks."""
    script = _PREAMBLE.format(root=REPO_ROOT) + f"""
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("onnxruntime", "torch"):
            raise ImportError("blocked for test: " + name)
        return None

sys.meta_path.insert(0, _Block())
sys.modules.pop("onnxruntime", None)

from src.rl.adapter import RLAdapter
a = RLAdapter({MODEL_PATH!r}, {os.path.join(REPO_ROOT, 'models', 'policy_meta.json')!r})
assert a.available is False, "adapter claimed available with onnxruntime blocked"
assert a.reason, "adapter unavailable without a reason"
print("ADAPTER_REASON:", a.reason)
try:
    a.step({{}}, {{}})
except RuntimeError as exc:
    print("STEP_RAISES:", exc)
else:
    raise AssertionError("step() on an unavailable adapter did not raise")

import src.main as m
assert m.ASVSystem is not None
print("MAIN_IMPORT_OK")

from src.control.controller_mux import ControllerMux
from src.control.thrust import ThrustComputer
from src.control.navigation import NavigationState, NavigationOutput
from src.sensors.fusion import FusedState
nav = NavigationOutput(mode=NavigationState.IDLE)
st = FusedState(gps_valid=True, imu_valid=True, mag_valid=True, heading=10.0)
mux = ControllerMux(ThrustComputer(), a, m.ASVSystem().navigator,
                    config={{"RL_ENABLED": False, "RL_DEFAULT_CONTROLLER": "pid"}})
cmd = mux.compute(nav, st)
assert mux.selected == "pid"
assert (cmd.surge, cmd.sway, cmd.yaw) == (0.0, 0.0, 0.0)
print("PID_TICK_OK", mux.selected)
"""
    proc = _run(script)
    assert proc.returncode == 0, f"child failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "ADAPTER_REASON:" in proc.stdout
    assert "MAIN_IMPORT_OK" in proc.stdout
    assert "PID_TICK_OK pid" in proc.stdout
    assert "STEP_RAISES: adapter unavailable" in proc.stdout


def test_model_missing_in_subprocess():
    """RL_MODEL_PATH points at a missing file: same fail-soft guarantees."""
    missing = os.path.join(REPO_ROOT, "models", "definitely_not_here.onnx")
    script = _PREAMBLE.format(root=REPO_ROOT) + f"""
from src.rl.adapter import RLAdapter
import src.config as cfg

a = RLAdapter({missing!r}, cfg.RL_META_PATH)
assert a.available is False, "adapter claimed available with no model file"
assert a.reason, "adapter unavailable without a reason"
assert "not found" in a.reason.lower() or "FileNotFound" in a.reason, a.reason
print("ADAPTER_REASON:", a.reason)
try:
    a.step({{}}, {{}})
except RuntimeError as exc:
    print("STEP_RAISES:", exc)
else:
    raise AssertionError("step() on an unavailable adapter did not raise")

# from_config() against the real config must also be fail-soft and silent-safe.
a2 = RLAdapter.from_config()
print("FROM_CONFIG:", a2.available, a2.reason)

import src.main as m
assert m.ASVSystem is not None
print("MAIN_IMPORT_OK")

from src.control.controller_mux import ControllerMux
from src.control.thrust import ThrustComputer
from src.control.navigation import NavigationState, NavigationOutput
from src.sensors.fusion import FusedState
nav = NavigationOutput(mode=NavigationState.WAYPOINT_NAV, desired_speed=0.5,
                       desired_heading=90.0)
st = FusedState(gps_valid=True, imu_valid=True, mag_valid=True, heading=0.0,
                latitude=-6.8969, longitude=112.0521)
mux = ControllerMux(ThrustComputer(), a, m.ASVSystem().navigator,
                    config={{"RL_ENABLED": False, "RL_DEFAULT_CONTROLLER": "pid"}})
cmd = mux.compute(nav, st)
assert mux.selected == "pid", mux.selected
assert cmd.surge > 0.0, "PID path produced no surge: %r" % (cmd,)
print("PID_TICK_OK", mux.selected)
"""
    proc = _run(script)
    assert proc.returncode == 0, f"child failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "ADAPTER_REASON:" in proc.stdout
    assert "MAIN_IMPORT_OK" in proc.stdout
    assert "PID_TICK_OK pid" in proc.stdout


def test_pid_path_ticks_in_this_process():
    """Belt-and-braces: on this box the mux ticks with the *real* fail-soft adapter."""
    from src.control.controller_mux import ControllerMux
    from src.control.navigation import NavigationOutput, NavigationState
    from src.control.thrust import ThrustComputer
    from src.rl.adapter import RLAdapter
    from src.sensors.fusion import FusedState

    adapter = RLAdapter.from_config()  # real config: the model is loadable, RL_ENABLED
    # does NOT gate the load (it is the mux's permission to *use* the policy). So the
    # adapter being available here is expected on a dev box; what must hold is that the
    # mux still selects PID because RL_ENABLED is False.
    assert adapter.available is True, adapter.reason

    nav = NavigationOutput(mode=NavigationState.WAYPOINT_NAV, desired_speed=0.5,
                           desired_heading=90.0)
    state = FusedState(gps_valid=True, imu_valid=True, mag_valid=True,
                       heading=0.0, latitude=-6.8969, longitude=112.0521)

    from src.main import ASVSystem
    mux = ControllerMux(ThrustComputer(), adapter, ASVSystem().navigator,
                        config={"RL_ENABLED": False, "RL_DEFAULT_CONTROLLER": "pid"})
    cmd = mux.compute(nav, state)
    assert mux.selected == "pid"
    assert mux.source == "pid"          # adapter available, still no policy
    assert cmd.surge > 0.0
