"""Measure the installed policy's action character — the numbers the tests below pin.

Run from pi-control with the venv python:

    .venv/Scripts/python.exe tools/describe_policy.py

Prints the case table, the grid saturation fraction and the a_prev lever for whatever
`models/policy.onnx` is currently vendored (see `models/MANIFEST.json` for provenance).
Read-only: writes nothing.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

META_PATH = os.path.join(ROOT, "models", "policy_meta.json")
MODEL_PATH = os.path.join(ROOT, "models", "policy.onnx")
_K = 10
A_PREV_ENV = (-0.992, -0.950, -0.738)


def act(session, meta, *, dx=0.0, dy=0.0, perr=0.0, u=0.0, v=0.0, r=0.0,
        a_prev=(0.0, 0.0, 0.0)):
    from src.rl.obs_np import normalize, raw_sample

    est = np.array([[0.0, 0.0, 0.0, u, v, r]], dtype=np.float64)
    goal = np.array([[dx, dy, perr, 0.0]], dtype=np.float64)
    s = normalize(raw_sample(est, goal, np.array([a_prev], dtype=np.float64)),
                  meta.get("obs_offset"), meta.get("obs_scale"),
                  meta.get("obs_clip_norm", 5.0))
    obs = np.tile(s, (1, _K)).astype(np.float32)
    out = session.run([meta.get("output_name", "action")],
                      {meta.get("input_name", "obs"): obs})[0]
    return np.asarray(out, dtype=np.float64).reshape(3)


def main():
    import onnxruntime as ort

    meta = json.load(open(META_PATH, encoding="utf-8"))
    man = json.load(open(os.path.join(ROOT, "models", "MANIFEST.json"), encoding="utf-8"))
    print(f"model   : {man['files']['models/policy.onnx']['src']}")
    print(f"ckpt    : {man['source_ckpt']}  steps={man['checkpoint_steps']}")
    s = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    print("\nA_PREV = zeros (episode first tick):")
    print("  %-20s %9s %9s %9s" % ("case", "surge", "sway", "yaw"))
    for name, kw in (("50 m ahead", dict(dx=50.0)),
                     ("50 m astern", dict(dx=-50.0, perr=np.pi)),
                     ("30 m starboard", dict(dy=30.0)),
                     ("30 m port", dict(dy=-30.0)),
                     ("0 m at goal", dict(dx=0.0, dy=0.0)),
                     ("hdg err 90", dict(dx=50.0, perr=np.deg2rad(90.0)))):
        a = act(s, meta, **kw)
        print("  %-20s %+9.4f %+9.4f %+9.4f" % (name, *a))

    print("\nA_PREV = env mid-episode %s:" % (A_PREV_ENV,))
    ahead = act(s, meta, dx=50.0, a_prev=A_PREV_ENV)
    astern = act(s, meta, dx=-50.0, perr=np.pi, a_prev=A_PREV_ENV)
    goal = act(s, meta, dx=0.0, dy=0.0, a_prev=A_PREV_ENV)
    stbd = act(s, meta, dy=30.0, a_prev=A_PREV_ENV)
    port = act(s, meta, dy=-30.0, a_prev=A_PREV_ENV)
    print("  ahead   %+.4f %+.4f %+.4f" % tuple(ahead))
    print("  astern  %+.4f %+.4f %+.4f" % tuple(astern))
    print("  stbd    %+.4f %+.4f %+.4f" % tuple(stbd))
    print("  port    %+.4f %+.4f %+.4f" % tuple(port))
    print("  at goal %+.4f %+.4f %+.4f" % tuple(goal))
    print(f"  goal sensitivity |astern-ahead| surge = {abs(astern[0]-ahead[0]):.4f}")
    print(f"  sway port/starboard split             = {abs(stbd[1]-port[1]):.4f}")

    grid = []
    for dx in (-70.0, -35.0, 0.0, 35.0, 70.0):
        for dy in (-70.0, -35.0, 0.0, 35.0, 70.0):
            for perr in (0.0, np.pi / 3, 2 * np.pi / 3, np.pi, -np.pi / 2):
                for u in (0.0, 0.3, 0.6, 1.0, 1.5):
                    grid.append(act(s, meta, dx=dx, dy=dy, perr=perr, u=u))
    A = np.asarray(grid)
    print(f"\ngrid n={len(A)}  surge [{A[:,0].min():+.4f}, {A[:,0].max():+.4f}]  "
          f"frac(surge>0.99)={float((A[:,0]>0.99).mean()):.3f}")
    print(f"  per-channel std = {A.std(axis=0).round(4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
