# P1 / T04 — vendor the DRL R7 policy runtime into pi-control

Owner: T04. Files written: `tools/sync_rl_runtime.py`, `src/rl/__init__.py`,
`src/rl/policy.py`, `src/rl/obs_np.py`, `src/rl/estimator.py`, `src/rl/frames.py`,
`src/rl/_vendor_note.py`, `models/policy.onnx`, `models/policy_meta.json`,
`models/MANIFEST.json`, this report. Nothing in `E:/buoy-project/drl` was touched.

## 1. What was copied

Source = DRL `E:/buoy-project/drl` @ `608256b224dc5e81dc541175df94256399fba4d4`.

| DRL source | PI target | bytes | sha256 |
|---|---|---|---|
| `runtime/policy.py` | `src/rl/policy.py` | 10733 | `3f6dc0c4dd6bc8e96d4a4d3c736aeb00de0ae2e9586ed136258c55d85f64c198` |
| `runtime/obs_np.py` | `src/rl/obs_np.py` | 5269 | `6b39e9a3f751027a5c24b39e9147825e937d6165e6902e5cf2e4c1be82fb2020` |
| `common/estimator.py` | `src/rl/estimator.py` | 3584 | `c09f23aca2aa56139442981554417c75f4227824a16f895f93cfdd1097bf2198` |
| `common/frames.py` | `src/rl/frames.py` | 4328 | `db1defb76e42cfebca86524fa7c91fa47735deb2349f2c5baca83509c3a9a441` |
| `runs/r7_policy.onnx` | `models/policy.onnx` | 356448 | `dbb90bb8b0123dc9276f0fefc3ff270937992a8fa30e8a14e617a8af41c49913` |
| `runs/r7_policy_meta.json` | `models/policy_meta.json` | 728 | `5bacf562fe4559e467e848d46cdc12a5064e1c8c56650dffa2144248c6399713` |

`models/MANIFEST.json` records, for every file above: `sha256`, `bytes`, and `src`
(DRL-relative path), plus `synced_utc`, `drl_commit`, `drl_path` (`"drl"`, relative to
`E:/buoy-project`), `checkpoint_steps` (299958272), `iteration` (4577),
`source_ckpt` (`runs/r7_300m/ckpt/ckpt_last.pt`) — the last three read from the R7 meta.

R7 is the newest checkpoint (`runs/policy.onnx` and `runs/policy_meta.json` are R6-era
and are never used).

## 2. Exact rewrites (the ONLY differences from DRL)

`diff` output, vendored file on the left, DRL original on the right:

```
--- src/rl/policy.py vs drl/runtime/policy.py
30,32c30,32
< from common.estimator import estimate_body_velocity
< from common.frames import heading_deg_to_psi, latlon_to_enu
< from runtime.obs_np import DEFAULT_CLIP_NORM, ObsBuilder
---
> from src.rl.estimator import estimate_body_velocity
> from src.rl.frames import heading_deg_to_psi, latlon_to_enu
> from src.rl.obs_np import DEFAULT_CLIP_NORM, ObsBuilder
--- src/rl/obs_np.py vs drl/runtime/obs_np.py      (see DEVIATION 1)
21c21
< from common.frames import goal_in_body, heading_error_rad
---
> from src.rl.frames import goal_in_body, heading_error_rad
--- src/rl/estimator.py vs drl/common/estimator.py
same
--- src/rl/frames.py vs drl/common/frames.py
same
```

## 3. Commands and real output

```
$ cd E:/buoy-project/pi-control
$ .venv/Scripts/python.exe tools/sync_rl_runtime.py          # run A (apply)
created   src/rl/policy.py
created   src/rl/obs_np.py
created   src/rl/estimator.py
created   src/rl/frames.py
created   src/rl/_vendor_note.py
created   models/policy.onnx
created   models/policy_meta.json
written models/MANIFEST.json

7 file(s) changed, 0 unchanged
exit=0

$ .venv/Scripts/python.exe tools/sync_rl_runtime.py          # run B (idempotency)
unchanged src/rl/policy.py
unchanged src/rl/obs_np.py
unchanged src/rl/estimator.py
unchanged src/rl/frames.py
unchanged src/rl/_vendor_note.py
unchanged models/policy.onnx
unchanged models/policy_meta.json
unchanged models/MANIFEST.json

0 file(s) changed, 7 unchanged
exit=0
   MANIFEST.json mtime/size/hash identical across A and B (byte-identical, synced_utc frozen)

$ .venv/Scripts/python.exe tools/sync_rl_runtime.py --check
unchanged src/rl/policy.py
... (all 7 unchanged)
in sync
exit=0
```

Line endings: Python files are copied **byte-for-byte**, so a CRLF source stays CRLF;
only the rewritten import lines differ (the replacement text carries no newline).
`.onnx` and the meta are copied as raw bytes (the R7 meta is CRLF JSON, preserved).

### Idempotency detail

`MANIFEST.json` is rewritten only when its content **excluding `synced_utc`** changes;
`synced_utc` is re-stamped only on an actual change. A file whose bytes already match is
never rewritten. The tool also never writes inside `--drl`.

### Drift detection (both paths proven, file restored afterwards)

```
# (a) one line appended to the vendored copy — src/rl/frames.py backed up first
$ .venv/Scripts/python.exe tools/sync_rl_runtime.py --check
unchanged src/rl/policy.py
unchanged src/rl/obs_np.py
unchanged src/rl/estimator.py
changed   src/rl/frames.py
unchanged src/rl/_vendor_note.py
unchanged models/policy.onnx
unchanged models/policy_meta.json

drift detected:
  src/rl/frames.py: changed (differs from rewrite of DRL source)
exit=1
# restored -> diff vs DRL: identical; --check -> "in sync", exit=0

# (b) manifest tampering (sha256 of policy.onnx + checkpoint_steps), backup restored
drift detected:
  models/MANIFEST.json: checkpoint_steps 1 != 299958272
  models/MANIFEST.json: stale entry for models/policy.onnx
exit=1
# restored -> "in sync", exit=0
```

### Missing-DRL path

```
$ .venv/Scripts/python.exe tools/sync_rl_runtime.py --drl E:/nonexistent/drl
error: DRL repo not found: E:\nonexistent\drl (pass --drl <path>)
exit=2
$ .venv/Scripts/python.exe tools/sync_rl_runtime.py --onnx runs/nope.onnx
error: missing in DRL repo: E:\buoy-project\drl\runs\nope.onnx
exit=2
```

One line on stderr, no traceback.

## 4. Vendored copy runs from the PI root, no torch

```
$ .venv/Scripts/python.exe -c "
import sys
from src.rl.policy import RLController
c = RLController('models/policy.onnx', 'models/policy_meta.json')
est = dict(gps_valid=True, imu_valid=True, mag_valid=True, sensor_ok=True, lat=-6.8969,
           lon=112.0521, heading_deg=0.0, speed_mps=0.0, course_deg=0.0, yaw_rate_deg_s=0.0)
goal = dict(lat=-6.8960, lon=112.0521, mode='waypoint')
print('action:', c.step(est, goal)); print('status:', c.status)
print('torch loaded:', 'torch' in sys.modules, '| onnxruntime loaded:', 'onnxruntime' in sys.modules)
print('safety:', c.safety, 'policy_dt_s:', c.policy_dt_s)"

action: (0.2, -0.2, 0.2)
status: {'source': 'policy', 'fault': None, 'action': (0.2, -0.2, 0.2)}
torch loaded: False | onnxruntime loaded: True
safety: {'clip': 1.0, 'rate_limit_per_s': 2.0, 'geofence_m': 80.0, 'watchdog_s': 0.5} policy_dt_s: 0.1
```

**The first step is slew-limited to 0.2 per axis by design**: `_emit` starts from
`_last_action = 0` and allows at most `rate_limit_per_s * policy_dt_s` =
2.0 × 0.1 = **0.2** change per axis on the first step (because `prev_t is None`,
`dt = policy_dt_s`). The `±0.2/0.2/0.2` above is therefore a limit, not the raw policy
output — the raw tanh output is larger. This is a deliberate handover protection, not a
fault; `source == "policy"`, `fault is None`.

## 5. `src/rl/__init__.py` import safety

Re-exports `RLController` eagerly (`src/rl/policy.py` imports onnxruntime only inside
`RLController.__init__`, so a bare import never needs it); `RLAdapter` and `RLLogger` are
resolved lazily via module `__getattr__` (PEP 562).

```
$ .venv/Scripts/python.exe -c "
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('onnxruntime','torch'): raise ImportError('blocked: '+name)
sys.meta_path.insert(0, Block())
import src.rl
print('src.rl OK ->', src.rl.RLController.__module__)
try: src.rl.RLAdapter
except BaseException as e: print('RLAdapter ->', type(e).__name__, e)"

src.rl OK -> src.rl.policy
exit=0
```

i.e. `import src.rl` survives even with **both** torch and onnxruntime blocked at
`sys.meta_path` level (`RLController` class attribute resolves, no onnxruntime import).

Regression check that nothing existing broke:
`.venv/Scripts/python.exe -m pytest test.py test_ml.py -q` → `50 passed in 1.79s`.

## 6. Deviations from the frozen interface

1. **`obs_np.py` needs one rewrite the interface said would not be needed.** The
   interface states `estimator.py` / `frames.py` / `obs_np.py` have "only their own
   relative-ish module imports if any (currently none)". `runtime/obs_np.py:21` does
   contain `from common.frames import goal_in_body, heading_error_rad`, and leaving it
   broke the vendored copy with `ModuleNotFoundError: No module named 'common'` on the
   first run. Fixed by applying the same documented mapping
   (`from common.frames import` → `from src.rl.frames import`) to `obs_np.py`; the tool,
   its docstring and `_vendor_note.py` all state this. No other rewrite was added, and
   `estimator.py`/`frames.py` are still byte-identical. **Tests that assert
   "obs_np.py is byte-identical" would fail — expected value must include this rewrite.**
2. `--check` prints the per-file `unchanged/changed/created` table like a normal run,
   then a `drift detected:` block with the diff list; exit 1. Missing DRL/model gives
   exit **2** (distinct from drift), one-line stderr message.
3. `drl_path` in the manifest is the DRL repo relative to the PI repo's parent
   (`"drl"`); when the two are not siblings it falls back to the absolute path.

## 7. Known issues carried forward (not fixed here, DRL is read-only)

Recorded per the brief, deliberately **not** patched in the vendored copy:

- `RLController.step` returns `np.zeros(3)` on a geofence breach **before** `_emit`, so a
  breach bypasses the slew limit (it is a step to zero, and `_last_action` is not
  updated to match).
- `_check` validates only `gps_valid` / `sensor_ok` / `gps None` / watchdog; a non-finite
  session output surfaces as `source == "zero"`, `fault == "nan"`.
- Consequence for the Pi side: the safety layer (ControllerMux) must do its own geofence
  (tighter, 20 m), latch itself, and treat `source in {"zero", "geofence"}` as a fault.

`--check` will flag any local modification of the vendored files, so fixing these in
`src/rl/` is not an option: the fix belongs in DRL or in the mux.

## 8. Re-running

```
cd E:/buoy-project/pi-control
.venv/Scripts/python.exe tools/sync_rl_runtime.py            # sync
.venv/Scripts/python.exe tools/sync_rl_runtime.py            # must print "0 file(s) changed"
.venv/Scripts/python.exe tools/sync_rl_runtime.py --check    # must exit 0
```
