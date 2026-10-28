"""Parity: the vendored RLController must be bit-equal to DRL's own copy.

Two independent checks:

1. **Action parity.** 200 pseudo-random synthetic ``(estimator_state, goal)``
   sequences are pushed through the vendored ``src.rl.policy.RLController`` and
   through DRL's ``runtime.policy.RLController`` loaded straight off the DRL
   repo, and every emitted action must agree within 1e-5.

2. **Artifact integrity.** Every sha256 in ``models/MANIFEST.json`` must match
   the file on disk, and the manifest's ``checkpoint_steps`` must equal the
   metadata's ``steps``.

DRL is read-only and may be absent here: the whole module skips with a clear
message when ``drl_path`` is not a directory (see ``conftest.drl_path``). The
``sys.path`` juggling needed to import ``runtime.*`` / ``common.*`` is confined
to ``_load_drl_controller`` below.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import pytest

from tests.conftest import MANIFEST_PATH, META_PATH, MODEL_PATH


# --------------------------------------------------------------------- helpers
class _StepClock:
    """Deterministic monotonic clock: one policy_dt per requested sample.

    ``RLController`` derives its slew-limit dt from consecutive ``clock()``
    calls, so both implementations must be driven from identical times or the
    comparison proves nothing.
    """

    def __init__(self, dt=0.1):
        self.dt = float(dt)
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self):
        self.t += self.dt


def _load_drl_controller(drl_path):
    """Import DRL's ``RLController`` with ``drl_path`` as the import root.

    ``runtime/`` and ``common/`` are top-level packages inside the DRL repo, so
    the repo root has to be on ``sys.path``. The path is removed again on the
    way out to keep this module's side effects as small as possible.
    """
    added = drl_path not in sys.path
    if added:
        sys.path.insert(0, drl_path)
    try:
        from runtime.policy import RLController  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - only on a broken DRL checkout
        pytest.skip(f"DRL runtime not importable from {drl_path}: {exc!r}")
    finally:
        if added and sys.path and sys.path[0] == drl_path:
            sys.path.pop(0)
    return RLController


def _session(model_path):
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def _estimate(rng: random.Random) -> dict:
    """A synthetic but physically-plausible estimator dict (CONTRACT §4.8)."""
    lat = -6.8969 + rng.uniform(-0.001, 0.001)
    lon = 112.0521 + rng.uniform(-0.001, 0.001)
    heading = rng.uniform(0.0, 360.0)
    speed = rng.choice([0.0, rng.uniform(0.0, 3.0)])
    gps_ok = rng.random() > 0.1
    imu_ok = rng.random() > 0.05
    mag_ok = rng.random() > 0.05
    return {
        "lat": lat, "lon": lon, "heading_deg": heading,
        "yaw_rate_deg_s": rng.uniform(-30.0, 30.0),
        "speed_mps": speed,
        "course_deg": (heading + rng.uniform(-20.0, 20.0)) % 360.0,
        "gps_valid": gps_ok, "imu_valid": imu_ok, "mag_valid": mag_ok,
        "sensor_ok": gps_ok and imu_ok and mag_ok,
    }


def _goal(rng: random.Random, est: dict) -> dict:
    """A synthetic goal: near the vessel (inside both geofences) or a hold point."""
    mode = rng.choice(["waypoint", "waypoint", "station_keeping"])
    near = rng.random() > 0.25
    if near:
        lat = est["lat"] + rng.uniform(-2e-4, 2e-4)
        lon = est["lon"] + rng.uniform(-2e-4, 2e-4)
    else:
        # ~70 m away: inside DRL's 80 m meta geofence, outside the mux's 20 m.
        lat = est["lat"] + rng.uniform(-6e-4, 6e-4)
        lon = est["lon"] + rng.uniform(-6e-4, 6e-4)
    goal = {"lat": lat, "lon": lon, "mode": mode}
    if rng.random() > 0.5:
        goal["heading_deg"] = rng.uniform(0.0, 360.0)
    if rng.random() > 0.7:
        goal["next_lat"] = lat + rng.uniform(-2e-4, 2e-4)
        goal["next_lon"] = lon + rng.uniform(-2e-4, 2e-4)
    if rng.random() > 0.8:
        goal["next2_lat"] = lat + rng.uniform(-2e-4, 2e-4)
        goal["next2_lon"] = lon + rng.uniform(-2e-4, 2e-4)
    return goal


# ----------------------------------------------------------------------- tests
def test_action_parity_200_sequences(drl_path, real_model_available):
    """200 random sequences: vendored action == DRL action within 1e-5."""
    if not real_model_available:
        pytest.skip("models/policy.onnx / policy_meta.json or onnxruntime missing")

    from src.rl.policy import RLController as VendoredController

    DRLCtrl = _load_drl_controller(drl_path)
    session_drl = _session(os.path.join(drl_path, "runs", "r7_policy.onnx"))
    session_vendor = _session(MODEL_PATH)

    clock_drl = _StepClock()
    clock_vendor = _StepClock()
    drl = DRLCtrl(onnx_path=os.path.join(drl_path, "runs", "r7_policy.onnx"),
                  meta_path=os.path.join(drl_path, "runs", "r7_policy_meta.json"),
                  session=session_drl, clock=clock_drl)
    vendored = VendoredController(onnx_path=MODEL_PATH, meta_path=META_PATH,
                                  session=session_vendor, clock=clock_vendor)

    rng = random.Random(20261028)
    checked = 0
    for i in range(200):
        est = _estimate(rng)
        goal = _goal(rng, est)
        a_drl = drl.step(dict(est), dict(goal))
        a_vendor = vendored.step(dict(est), dict(goal))
        assert len(a_drl) == 3 and len(a_vendor) == 3
        for axis, (d, v) in enumerate(zip(a_drl, a_vendor)):
            assert abs(d - v) <= 1e-5, (
                f"sequence {i} axis {axis}: DRL={d!r} vendored={v!r} "
                f"delta={abs(d - v):.3e}\n est={est}\n goal={goal}\n"
                f" status_drl={drl.status} status_vendored={vendored.status}")
        # The status dicts must agree too (source + fault wording is frozen).
        assert drl.status["source"] == vendored.status["source"], (
            f"sequence {i}: source {drl.status['source']!r} != "
            f"{vendored.status['source']!r}")
        assert drl.status["fault"] == vendored.status["fault"], (
            f"sequence {i}: fault {drl.status['fault']!r} != "
            f"{vendored.status['fault']!r}")
        checked += 1
        clock_drl.advance()
        clock_vendor.advance()

    assert checked == 200


def test_manifest_hashes_match_disk():
    """Every sha256 in MANIFEST.json matches the file; steps match the meta."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest["files"]
    assert files, "MANIFEST.json has no files entries"
    repo_root = os.path.dirname(os.path.dirname(MANIFEST_PATH))
    for rel_path, entry in files.items():
        full = os.path.join(repo_root, rel_path.replace("/", os.sep))
        assert os.path.isfile(full), f"manifest lists a missing file: {rel_path}"
        with open(full, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        assert digest == entry["sha256"], (
            f"{rel_path}: on-disk sha256 {digest} != manifest {entry['sha256']}")
        assert os.path.getsize(full) == entry["bytes"], (
            f"{rel_path}: size {os.path.getsize(full)} != manifest {entry['bytes']}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert manifest["checkpoint_steps"] == meta["steps"], (
        f"manifest checkpoint_steps {manifest['checkpoint_steps']} != "
        f"meta steps {meta['steps']}")


def test_vendored_obs_np_is_rewritten_copy(drl_path):
    """The vendored package must differ from DRL *only* by the import rewrite.

    Guards the sync contract: if someone hand-edits ``src/rl/*`` the copy drifts
    from DRL and ``--check`` would fail. Uses the DRL source when present.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drift = []
    expected = {
        "src/rl/estimator.py": "common/estimator.py",
        "src/rl/frames.py": "common/frames.py",
    }
    for vend_rel, drl_rel in expected.items():
        vend = os.path.join(repo_root, *vend_rel.split("/"))
        src = os.path.join(drl_path, *drl_rel.split("/"))
        if not (os.path.isfile(vend) and os.path.isfile(src)):
            continue
        with open(vend, "rb") as f1, open(src, "rb") as f2:
            if f1.read() != f2.read():
                drift.append(vend_rel)
    assert not drift, f"vendored files differ from DRL byte-for-byte: {drift}"
