"""Inference latency bench for the Pi-side RLAdapter.

    .venv/Scripts/python.exe tools/bench_inference.py --n 1000
    python tools/bench_inference.py --n 1000 --model models/policy.onnx --meta models/policy_meta.json

Feeds ``--n`` synthetic estimator states (a fixed 150-float observation is what the
vendor builds internally; the inputs here give a real, non-trivial obs: lat/lon creeping
east+north, heading 45 deg, 1.5 m/s at 45 deg COG) through the **real** ``RLAdapter`` and
its onnxruntime ``CPUExecutionProvider`` session, ``intra_op_num_threads=1``, and prints
p50/p95/max of the wall time per ``adapter.step()`` in ms.

The number is a **laptop x86** number. It is NOT the Pi 3B number: measure on the Pi with
the same command before trusting any ``RL_MAX_INFERENCE_MS`` budget there.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.adapter import RLAdapter  # noqa: E402


def synth_est(i: int) -> dict:
    """Estimator dict per CONTRACT §4.8, drifting slowly so the obs keeps changing."""
    t = i * 0.1
    return {
        "lat": 10.0 + t * 1e-6,
        "lon": 106.0 + t * 1e-6,
        "heading_deg": 45.0,
        "yaw_rate_deg_s": 2.0,
        "speed_mps": 1.5,
        "course_deg": 45.0,
        "gps_valid": True,
        "imu_valid": True,
        "mag_valid": True,
        "t": time.monotonic(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1000, help="number of steps (default 1000)")
    ap.add_argument("--warmup", type=int, default=20, help="untimed warmup steps")
    ap.add_argument("--model", default=None, help="default: src.config RL_MODEL_PATH, "
                     "else <repo>/models/policy.onnx when the configured path is absent")
    ap.add_argument("--meta", default=None, help="same fallback as --model")
    args = ap.parse_args(argv)

    # src.config paths point at the Pi runtime dir (~/pi-controller); on a dev box the
    # model to bench is the one in this repo. Fall back only when the configured file
    # is missing, but report which path was used.
    model, meta = args.model, args.meta
    if model is None:
        cfg = _config()
        cfg_model = getattr(cfg, "RL_MODEL_PATH", None)
        cfg_meta = getattr(cfg, "RL_META_PATH", None)
        if cfg_model and Path(cfg_model).is_file():
            model, meta = cfg_model, cfg_meta
        else:
            model, meta = ROOT / "models" / "policy.onnx", ROOT / "models" / "policy_meta.json"
            print(f"note: RL_MODEL_PATH={cfg_model!r} absent, benching {model}")
    # Bench measures the artifact, not the deployment switch RL_ENABLED (False by
    # design; the mux is what honours it).
    adapter = RLAdapter(model, meta, enabled=True)
    if not adapter.available:
        print(f"adapter unavailable: {adapter.reason}")
        return 2
    print(f"model={adapter.model_path}")
    print(f"onnxruntime={_ort_version()}  policy_dt_s={adapter.policy_dt_s}  n={args.n}")

    goal = {"lat": 10.0002, "lon": 106.0002, "heading_deg": 45.0, "mode": "waypoint"}
    for i in range(args.warmup):
        adapter.step(synth_est(i), goal)
    adapter.reset()

    samples = []
    t0 = time.perf_counter()
    for i in range(args.n):
        adapter.step(synth_est(i), goal)
        samples.append(adapter.last_inference_ms)
    wall = time.perf_counter() - t0

    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    print(f"last action={adapter.status.get('action')} source={adapter.status.get('source')}")
    print(f"inference ms: p50={p50:.3f}  p95={p95:.3f}  max={samples[-1]:.3f}  "
          f"mean={wall / args.n * 1e3:.3f}  (n={args.n}, warmup={args.warmup})")
    print("NOTE: host = this machine (x86 laptop). NOT the Pi 3B; measure there.")
    return 0


def _config():
    try:
        from src import config
        return config
    except Exception:
        return None


def _ort_version() -> str:
    try:
        import onnxruntime as ort
        return ort.__version__
    except Exception as exc:  # pragma: no cover
        return f"? ({exc})"


if __name__ == "__main__":
    raise SystemExit(main())
