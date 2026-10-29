# P1_T05 — Pi-side RLAdapter + aarch64 onnxruntime pin + bench tool

Owner: T05. Files: `src/rl/adapter.py`, `tools/bench_inference.py`, `requirements.txt`
(onnxruntime line only), this report. Nothing else touched.

Environment: `E:/buoy-project/pi-control`, runner `.venv/Scripts/python.exe`
(Python 3.13.3, onnxruntime 1.31.1, numpy 2.5.4). Model **R7**
(`models/policy.onnx` = copy of `drl/runs/r7_policy.onnx`,
sha256 `dbb90bb8b0123dc9276f0fefc3ff270937992a8fa30e8a14e617a8af41c49913`,
matches `models/MANIFEST.json`; DRL commit `608256b224dc5e81dc541175df94256399fba4d4`).

---

## 1. Interface (as implemented)

```python
@dataclass(frozen=True)
class PolicyResult:
    action: Tuple[float, float, float]      # post-clip + post-slew, from RLController
    raw_action: Tuple[float, float, float]  # == action (see raw_action note)
    source: str                             # "policy" | "fallback" | "geofence" | "zero" | "init"
    fault: Optional[str]                    # None | "gps" | "sensor" | "watchdog" | "nan" | ...
    inference_ms: float                     # wall time of THIS call

class RLAdapter:
    def __init__(self, model_path=None, meta_path=None, *, enabled=True, log=None,
                 intra_op_num_threads=1, session=None, safety=None, clock=time.monotonic)
    available: bool
    reason: str | None
    last_inference_ms: float                 # 0.0 before the first step and after reset()
    meta: dict
    policy_dt_s: float                       # meta, 0.1 fallback
    model_path / meta_path: str | None       # resolved path is exposed for the report/bench
    def reset() -> None
    def step(est: dict, goal: dict) -> PolicyResult   # RuntimeError only when unavailable
    status: dict                                     # RLController.status passthrough
    @classmethod
    def from_config(cls, cfg=None, **kw) -> RLAdapter
```

Deviations from the frozen text, all additive:

1. `clock` is accepted (default `time.monotonic`) and forwarded to `RLController`, so
   parity/replay tests can drive both sides from the same clock. `last_inference_ms` uses
   `time.perf_counter`, never the injected clock (it is a wall-time measurement).
2. `model_path` / `meta_path` are kept as attributes after construction (the freeze did not
   forbid extra fields; the bench and the report need the resolved path).
3. `raw_action` is documented and implemented as `== action`. Reason: `RLController.step`
   returns the post-`_emit` tuple and keeps the pre-slew session output in a local
   `action` variable inside `_policy`, so the adapter cannot obtain it without editing
   vendored code (forbidden). The freeze explicitly allows this with documentation.
4. `enabled=False` sets `available=False`/`reason="disabled"` **without** a warning — an
   explicit deployment switch is not a failure, and `from_config()` on the real config
   (`RL_ENABLED = False`) must stay silent. `reason="disabled"` is distinguishable from
   every genuine failure reason.
5. `from_config()` reads `RL_MODEL_PATH`, `RL_META_PATH` (both `getattr(..., None)`;
   additionally tolerates a **dict** config for tests) and `RL_ENABLED` (default `True`
   when the name is absent). Every one can be overridden by keyword (`enabled=`, `safety=`,
   `log=`), which is what `tools/bench_inference.py` uses to bench the artifact regardless
   of the deployment switch.
6. Internal building is factored into `_build()`, so `__init__` has one `try/except
   Exception` and one warning.

Session options: `intra_op_num_threads=RLAdapter(intra_op_num_threads)` (default 1),
`inter_op_num_threads=1`, provider list exactly `["CPUExecutionProvider"]`.

`step()` holds one `threading.Lock` for the duration of the step (session run + status
copy); nothing else takes that lock. `reset()` takes it too. No file I/O, no sleeping.

---

## 2. Fail-soft matrix — actually executed

Script: `C:/Users/AORUS/AppData/Local/hermes/cache/scratch/t05_smoke.py`
(run `cd E:/buoy-project/pi-control && .venv/Scripts/python.exe <script>`), 34 checks,
final line `RESULT: ALL PASS`, exit 0. Full log:
`.../scratch/t05_smoke.txt`.

| # | injected failure | result | observed `reason` |
|---|---|---|---|
| 1 | `model_path` points at a non-existent file | `available=False`, no raise | `FileNotFoundError: model not found: ...\nope.onnx` |
| 1b | `step()` on the unavailable adapter | raises `RuntimeError("adapter unavailable: <reason>")` | — |
| 2 | no `model_path`/`meta_path` at all | `available=False` | `ValueError: no model_path` |
| 3 | meta file is invalid JSON (`{not json`) | `available=False` | `JSONDecodeError: Expecting property name ...` |
| 4 | model file exists but is not ONNX (`b"not an onnx file"`) | `available=False` | `InvalidProtobuf: [ONNXRuntimeError] : 7 : INVALID_PROTOBUF ...` |
| 5 | `onnxruntime` not importable (`sys.modules["onnxruntime"] = None`) | `available=False`, valid model+meta present | `ModuleNotFoundError: import of onnxruntime halted; None in sys.modules` |
| 6 | `src.rl.policy` not importable (`sys.meta_path` blocker raising `ImportError`) | `available=False`, session built OK first | `ImportError: blocked for test` |
| 7 | `enabled=False` | `available=False`, **zero** warning records | `disabled` |
| 8 | `from_config(SomeClassWithoutRLNames)` | no raise, `available=False` | `ValueError: no model_path` |
| 9 | `from_config({"RL_MODEL_PATH":..., "RL_ENABLED": True})` | `available=True` | — |
| 10 | `from_config()` against the real `src.config` (`RL_ENABLED = False`) | `available=False`, silent | `disabled` |

Exactly **7** `WARNING` records were emitted by `src.rl.adapter` for cases 1,2,3,4,5,6,8
(one per genuine failure, none for case 7/10). Missing `onnxruntime` and missing vendored
policy are therefore both fail-soft, and neither prints to stdout/stderr — verified with a
bare `import src.rl.adapter` run.

### 2b. Injected fake session (no onnxruntime, no real model)

| check | result |
|---|---|
| `session=` injection -> `available=True`, no `onnxruntime` needed | PASS |
| `policy_dt_s == 0.1` read from meta | PASS |
| `last_inference_ms == 0.0` before the first step | PASS |
| first step: session output `(0.9,-0.5,0.2)` -> applied `(0.2,-0.2,0.2)` (slew limit 2.0/s × 0.1 s) | PASS |
| `source="policy"`, `fault=None`, `raw_action == action` | PASS |
| `inference_ms == last_inference_ms > 0` | PASS |
| fake session received a `[1,150]` float obs | PASS |
| `reset()` -> `last_inference_ms == 0.0`, `status["source"] == "init"` | PASS |
| `gps_valid=False` step -> `source="fallback"`, `fault="gps"` | PASS |
| `status` dict has `{"source","fault","action"}` when unavailable | PASS |

### 2c. Real path (model + vendored runtime present)

```
real step action: (0.2033777999924496, 0.2033777999924496, 0.2033777999924496) source: policy ms: 0.123
legacy action:  (0.2, -0.2, 0.2)
adapter action: (0.2, -0.2, 0.2)
PASS adapter == legacy RLController on first step
```

The "legacy" number comes from `drl/runtime/policy.py::RLController` loaded straight off the
DRL repo with `r7_policy.onnx` + `r7_policy_meta.json` (the T04 parity one-liner, same
`est`/`goal`), driven from `lat=10.0, lon=106.0, heading_deg=45.0, yaw_rate=0.0,
speed_mps=0.0, course_deg=45.0, all-valid`, `goal = lat 10.0002, lon 106.0002,
heading_deg 45.0, mode "waypoint"`. Bit-identical to 1e-6 — the adapter changes nothing.
(The second loop line above uses a moving est, hence the larger applied action.)

---

## 3. aarch64 onnxruntime pin

### 3.1 What was pinned

`requirements.txt` gained exactly one dependency line plus a 4-line comment, inserted after
`scikit-learn>=1.3.0` under the existing `# ML inference` heading; nothing else in the file
changed, all existing comments kept, encoding/EOL untouched (file is CRLF in the worktree
and, per `git config core.autocrlf=true`, LF in the index — the added block follows both).
`git diff --stat` = `5 insertions(+)`, one hunk.

```
# PPO policy runtime (RL_* in src/config.py). This is the Raspberry Pi 3B /
# Raspberry Pi OS Lite 64-bit (bookworm, glibc 2.36) aarch64 wheel:
# onnxruntime-<ver>-cp312-cp312-manylinux_2_28_aarch64.whl (also cp311/cp313).
# Deliberately NO torch: inference runs on onnxruntime+numpy only.
onnxruntime==1.31.1
```

**Pin = `onnxruntime==1.31.1`.**

- Why 1.31.1 and not the newest-ranged/loosest choice: 1.31.1 is the newest release and the
  only one probed whose cp312 aarch64 wheel is tagged **`manylinux_2_28_aarch64`** without
  the older `manylinux_2_27` fallback tag; it is also the version already verified on this
  box against R7 (brief §"Verified working") and already installed in `.venv`. 1.24.4 and
  below carry `manylinux_2_27_aarch64.manylinux_2_28_aarch64` (both fine on bookworm, but
  the double tag means they also target glibc 2.27 distros). 1.31.1 therefore needs
  **glibc >= 2.28** — Pi OS Lite 64-bit **bookworm ships glibc 2.36**, so the wheel is
  satisfied with a large margin. `bullseye` (glibc 2.31) would also satisfy 2.28; anything
  older than glibc 2.28 (buster, glibc 2.28 exactly is the floor) would not.
- **torch is deliberately not a dependency**: the policy is exported to ONNX, inference
  runs on `onnxruntime` + `numpy` only. Torch on a Pi 3B aarch64 is a multi-hundred-MB
  wheel with no armv7/aarch64 CPU wheel for every Python version and is far too slow/heavy
  for 10 Hz inference on 4 shared cores.

### 3.2 Reproduced orchestrator finding (manylinux2014_aarch64 caps at <=1.16.3)

```
$ .venv/Scripts/python.exe -m pip download --no-deps --only-binary=:all: \
    --platform manylinux2014_aarch64 --python-version 3.11 --implementation cp --abi cp311 \
    "onnxruntime>=1.17" -d C:/Users/AORUS/AppData/Local/hermes/cache/scratch/ortprobe_t05
ERROR: Could not find a version that satisfies the requirement onnxruntime>=1.17
       (from versions: 1.15.0, 1.15.1, 1.16.0, 1.16.1, 1.16.2, 1.16.3)
ERROR: No matching distribution found for onnxruntime>=1.17
```

Confirmed: `manylinux2014_aarch64` resolves nothing newer than **1.16.3**. The modern tag is
`manylinux_2_27`/`manylinux_2_28`, so `manylinux_2_28_aarch64` is the platform to test.

### 3.3 Exact required commands and output

```
$ .venv/Scripts/python.exe -m pip download --no-deps --only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 3.12 --implementation cp --abi cp312 onnxruntime==1.31.1 -d C:/Users/AORUS/AppData/Local/hermes/cache/scratch/ortprobe_t05
Collecting onnxruntime==1.31.1
  Using cached onnxruntime-1.31.1-cp312-cp312-manylinux_2_28_aarch64.whl.metadata (5.8 kB)
Using cached onnxruntime-1.31.1-cp312-cp312-manylinux_2_28_aarch64.whl (21.4 MB)
Saved c:\users\aorus\appdata\local\hermes\cache\scratch\ortprobe_t05\onnxruntime-1.31.1-cp312-cp312-manylinux_2_28_aarch64.whl
Successfully downloaded onnxruntime

$ .venv/Scripts/python.exe -m pip download --no-deps --only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 3.11 --implementation cp --abi cp311 onnxruntime==1.31.1 -d C:/Users/AORUS/AppData/Local/hermes/cache/scratch/ortprobe_t05
Collecting onnxruntime==1.31.1
  Using cached onnxruntime-1.31.1-cp311-cp311-manylinux_2_28_aarch64.whl.metadata (5.8 kB)
Using cached onnxruntime-1.31.1-cp311-cp311-manylinux_2_28_aarch64.whl (21.4 MB)
Saved c:\users\aorus\appdata\local\hermes\cache\scratch\ortprobe_t05\onnxruntime-1.31.1-cp311-cp311-manylinux_2_28_aarch64.whl
Successfully downloaded onnxruntime
```

Wheel files actually on disk afterwards:

```
-rw-r--r-- 1 AORUS 197121 21413526 onnxruntime-1.31.1-cp311-cp311-manylinux_2_28_aarch64.whl
-rw-r--r-- 1 AORUS 197121 21420803 onnxruntime-1.31.1-cp312-cp312-manylinux_2_28_aarch64.whl
```

Extra probes (all with `--platform manylinux_2_28_aarch64 --only-binary=:all: --no-deps`):

| version | py 3.11 / cp311 | py 3.12 / cp312 | py 3.13 / cp313 |
|---|---|---|---|
| 1.31.1 | `cp311-cp311-manylinux_2_28_aarch64` | `cp312-cp312-manylinux_2_28_aarch64` | `cp313-cp313-manylinux_2_28_aarch64` |
| 1.24.4 | `cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64` | same, cp312 | same, cp313 |
| 1.23.2 | `cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64` | same, cp312 | same, cp313 |
| 1.22.1 | n/a | `cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64` | n/a |
| 1.20.1 | n/a | `cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64` | n/a |

- 1.31.1 exists for cp311 **and** cp312 **and** cp313 on aarch64 (older 1.x had no cp313
  wheels at all — 1.24.4 is the first with cp313). Pin stays usable if the Pi moves from
  bookworm's Python 3.11 to a 3.12 or 3.13 interpreter.
- These are **resolution** proofs (pip accepted the wheel for that platform/ABI). They are
  not an install proof on a Pi — nothing aarch64 was executed here.

### 3.4 Pi OS / glibc note

Manylinux tag -> required glibc floor:

| tag | glibc floor | Pi OS release (glibc) |
|---|---|---|
| `manylinux2014_aarch64` | 2.17 | buster 2.28, bullseye 2.31, bookworm 2.36 |
| `manylinux_2_27_aarch64` | 2.27 | bullseye 2.31, bookworm 2.36 |
| `manylinux_2_28_aarch64` | **2.28** | **bullseye 2.31 OK, bookworm 2.36 OK** |

`onnxruntime==1.31.1`'s aarch64 wheel is `manylinux_2_28`, so it installs on
**Raspberry Pi OS Lite 64-bit (bookworm) = glibc 2.36** and on bullseye (2.31), and would
fail on buster (2.28 is the floor, buster is exactly 2.28 — borderline, do not rely on it).
The Pi 3B is 64-bit capable (Cortex-A53), so the 64-bit Lite image is the assumption stated
in `requirements.txt`; on a 32-bit (armv7) image **no** onnxruntime wheel exists at all.

---

## 4. Bench

`tools/bench_inference.py` — `--n 1000`, 20 untimed warmups, synthetic estimator dicts per
step (lat/lon creeping at 0.1 s intervals, heading 45°, yaw rate 2 °/s, SOG 1.5 m/s, COG 45°,
all-valid), 150-float obs built by the vendored `ObsBuilder`, real `RLAdapter` and real
`onnxruntime` session. `--warmup` (default 20) and `--model/--meta` are the only extra flags.
Model resolution: `src.config.RL_MODEL_PATH` when that file exists, else
`<repo>/models/policy.onnx` with a printed note (`src/config.py` points at the Pi runtime
path `~/pi-controller/models/...`, which does not exist on this dev box).

Exact command and output, run for real:

```
$ cd E:/buoy-project/pi-control
$ .venv/Scripts/python.exe tools/bench_inference.py --n 1000
note: RL_MODEL_PATH='C:\\Users\\AORUS\\pi-controller\\models\\policy.onnx' absent, benching E:\buoy-project\pi-control\models\policy.onnx
model=E:\buoy-project\pi-control\models\policy.onnx
onnxruntime=1.31.1  policy_dt_s=0.1  n=1000
last action=(0.4540423999889754, 0.4583961606549565, 0.5095425248146057) source=policy
inference ms: p50=0.143  p95=0.244  max=0.465  mean=0.170  (n=1000, warmup=20)
NOTE: host = this machine (x86 laptop). NOT the Pi 3B; measure there.
```

**This is a laptop x86 number** (AMD64, Windows 11, CPython 3.13, `intra_op_num_threads=1`).
It is **NOT** the Pi 3B number: a Pi 3B Cortex-A53 at 1.2 GHz is roughly 10–20× slower per
core for this kind of single-thread GEMM, and 4 slow cores are shared with the 20 Hz sensor
fusion/telemetry threads. The Pi number **must be measured on the Pi with this exact
command** before `RL_MAX_INFERENCE_MS = 50` (the value T03 put in `src/config.py`) is treated
as validated. The measured p50 on a laptop leaves ~300× headroom against that budget; the
mux's `slow_inference` branch is therefore untested against real hardware.

---

## 5. Regression / integration checks run

| command | result |
|---|---|
| `.venv/Scripts/python.exe -m py_compile src/rl/adapter.py tools/bench_inference.py` | exit 0 |
| `.venv/Scripts/python.exe tools/bench_inference.py --n 1000` | exit 0, output in §4 |
| `.venv/Scripts/python.exe <scratch>/t05_smoke.py` | exit 0, `RESULT: ALL PASS` (34 checks) |
| `.venv/Scripts/python.exe -c "from hardware_mock import install; install(); import src.rl.adapter"` | exit 0, **no** warning/print from the adapter (import-safe, `RL_ENABLED=False` silent) |
| `.venv/Scripts/python.exe -m pytest test.py test_ml.py -q` | `50 passed in 1.43s` (baseline preserved) |
| `.venv/Scripts/python.exe -m pytest -q` | `33 passed` (only the 3 test files that existed at the time: `test.py`, `test_ml.py`, T10's `tests/…`) |

---

## 6. Open problems / notes for the parent

1. **`raw_action == action`** — see §1.3. The mux/log row will therefore show identical
   raw and applied surge/sway/yaw, except for `RL_MAX_THRUST_SCALE` (which the mux applies
   after the adapter) and the mux's own handover slew. If a true pre-slew value is wanted,
   it needs an export change in DRL or a vendored-code change — neither is owned by T05.
2. **Known DRL bug carried into the vendored copy** (per brief, do not fix): a geofence
   breach in `RLController.step` returns `np.zeros(3)` **before** `_emit`, so it bypasses the
   slew limit, and a non-finite session output becomes `source="zero"`, `fault="nan"`. The
   adapter passes `source`/`fault` through unchanged, which is what the frozen `ControllerMux`
   contract expects (`treat source in {"zero","geofence"} as a fault for latching`). No
   compensating logic was added on the adapter side — the mux must own the tighter 20 m
   geofence and the latch.
3. **Pi latency unmeasured** — §4. Also unmeasured on the Pi: import time of
   `onnxruntime` on aarch64 (a few hundred ms, first-boot cost) and RSS of the session.
4. **`manylinux_2_28` floor** — if the deployment image is 32-bit Raspberry Pi OS, or
   buster, the pin has no wheel. Recorded in `requirements.txt` as the 64-bit bookworm
   assumption.
5. **Session reuse on a fault** — `RLController` resets its obs ring buffer whenever
   `source != "policy"`, so the first step after a recovery is a degenerate 1-sample
   history. That is upstream behaviour, unchanged here; if the policy visibly misbehaves
   right after a GPS blip the mux should count those steps as warm-up rather than judge them.
6. `src/config.py` became available mid-task (T03 landed `RL_*`, all `BASE_DIR`-relative),
   so `RLAdapter.from_config()` was verified against the real config, not only the
   `getattr`-default path: `available=False`, `reason="disabled"`, silent. The default-safe
   path (`from_config` on a config object without any `RL_*` name) is still covered by §2
   case 8.
7. `tools/bench_inference.py` falls back to `<repo>/models/policy.onnx` when
   `src.config.RL_MODEL_PATH` does not exist **on the dev box**; on the Pi the configured
   path exists and is used. If a future reviewer prefers strict "config path or fail", the
   fallback is one `if` in `main()`.
