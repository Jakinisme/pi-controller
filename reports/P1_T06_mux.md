# P1 / T06 — ControllerMux (safety layer around the trained policy)

Owner: T06. Files: `src/control/controller_mux.py` (new), this report. Nothing else
touched (no git state change; only the orchestrator commits).

Environment: `E:/buoy-project/pi-control`, runner `.venv/Scripts/python.exe`
(Python 3.13.3, onnxruntime 1.31.1, numpy 2.5.4). Model **R7**
(`models/policy.onnx` + `models/policy_meta.json`).

---

## 1. Interface (as implemented)

```python
class ControllerMux:
    def __init__(self, thrust, adapter, navigator, gps=None, *, config=None,
                 clock=time.monotonic, logger=None, rl_logger=None)
    def compute(self, nav_output, state, now=None) -> ThrustCommand
    def set_controller_request(self, name) -> None   # "pid" | "ppo"
    def reset(self) -> None
    source: str            # "pid" | "policy" | "fallback" | "geofence" | "zero"
    fault: Optional[str]   # None | "gps" | "imu" | "mag" | "watchdog" |
                           #   "inference" | "nan" | "slow_inference" |
                           #   "unavailable" | "geofence" | "latched"
    selected: str          # "pid" | "ppo"
    rl_logger              # injected RLLogger (or anything with .log(row))
```

`ThrustCommand`, `FusedState`, `NavigationState` and `NavigationOutput` are
**imported**, never duplicated. `compute` returns the same
`ThrustCommand(surge, sway, yaw, timestamp)` shape as `ThrustComputer.compute`.

Additive attribute: none needed beyond the freeze except a private `_rl_active`
flag (see §3.3).

---

## 2. Watchdog clock decision (required call-out)

`src/sensors/fusion.py` sets `FusedState.timestamp = time.time()` — a **wall-clock**
stamp, **not** a monotonic sample time. It therefore cannot be subtracted from the
injected monotonic `clock()` used by the rest of the control loop (and by
`RLController`). Decision, documented in the module docstring and repeated here:

> The mux keeps its **own** sensor-freshness timestamp. It records `clock()` every
> time the fused state it is handed **changes** (any of `timestamp, latitude,
> longitude, heading, speed, yaw_rate, gps_valid, imu_valid, mag_valid` differs
> from the previous tick). A state that has not changed for longer than
> `RL_WATCHDOG_S` (0.5 s) is a **watchdog** fault → `source="fallback"`,
> `fault="watchdog"`, PID for the tick.

Keyed on **values, never `id(state)`**: CPython reuses object addresses after GC, so
an id-based key false-negatives when the caller builds a fresh `FusedState` per tick
(the real `src/main.py` loop reuses one object, but tests and the mock do not). An
unchanged sample means a stale sample — value keying is both correct and id-reuse safe.

This does **not** double-count the policy's own internal watchdog
(`RLController._check` compares the estimator `t` it is handed against its own
`clock()`), which still runs and surfaces its own `fault` through `PolicyResult`.
The mux adds one independent freshness gate on top; when *its* check fires it emits
`source="fallback"`, `fault="watchdog"` itself.

---

## 3. Frozen rules — how each is implemented

1. **Selection.** `selected=="ppo"` requires `RL_ENABLED is True` **and**
   (`_request=="ppo"` or `RL_DEFAULT_CONTROLLER=="ppo"`) **and**
   `adapter.available`. Otherwise `"pid"`. Resolved every tick in `_resolve_selected`.
   `RL_ENABLED=False` (the shipped default) therefore keeps PID even if an operator
   sends `ppo` — but the operator can still raise `RL_ENABLED` at deploy time.
2. **Mode gate.** Only `WAYPOINT_NAV` and `STATION_KEEP` may run the policy. `IDLE`
   and `MANUAL` are a direct `ThrustComputer.compute(nav_output, state)` pass-through,
   byte-identical and **never slew-limited** (rule 6), within one tick.
3. **Decimation.** `adapter.step` runs when `now - last_step_t >= policy_dt_s * 0.9`
   (first tick always steps, since `last_step_t is None`). In-between ticks return a
   `ThrustCommand` carrying the last applied policy action — no inference. Jitter:
   measured gap >30 % off `policy_dt_s` → one `log.warning` per 5 s containing the
   gap in ms.
4. **Fault ladder** (first match wins, every tick):
   `gps` → `imu` → `mag` → `watchdog` → `inference` (step raised) → `nan` (non-finite)
   → `slow_inference` (`last_inference_ms > RL_MAX_INFERENCE_MS`), each → PID for the
   tick with the matching `source="fallback"`/`fault`. Geofence (row 8, `zero` +
   `navigator.stop()` + latch) is evaluated **after** the sensor gates and **before**
   the policy runs, satisfying rule 5's "tighter and runs first" — a sensor fault
   outranks a geofence, which is the ladder's own ordering.
   **Latch.** 3 consecutive fallback/nan/slow_inference faults → latched; geofence
   latches immediately. A latched tick behaves as PID and reports
   `source="fallback"`, `fault="latched"`. Cleared only by `set_controller_request`
   or `reset()` (a new command). `unavailable` while RL enabled+requested also latches.
5. **Geofence.** ENU origin = first valid fix of the current RL session
   (`self._lat0/_lon0`, set on the first policy-eligible tick). Distance via
   `src.utils.geo.haversine_distance` (the same function the rest of the repo uses)
   > `RL_GEOFENCE_M` (20.0) → zero thrust + `navigator.stop()` + latch. The
   controller's own 80 m meta geofence is **not** the one that matters; the mux one is
   checked first and is tighter.
6. **Handover slew.** `rate_limit_per_s` (read from `adapter.safety`, falling back to
   `adapter.meta["safety"]`, finally 2.0) × `policy_dt_s` = 0.2 per step. Applied to
   every emitted autonomous command (policy, fallback, zero, and the PID tick that
   follows a policy tick). **Never** applied to the `IDLE`/`MANUAL` pass-through.
   *PID-only deployments are byte-exact*: slew engages only once a policy command has
   actually been emitted (`_rl_active`), so a pure-PID run — or the very first PID
   tick before any policy output — equals `ThrustComputer.compute` exactly (rule 2).
7. **`RL_MAX_THRUST_SCALE`** multiplies the **policy** action only, clipped to ±1
   after scaling, recorded in the log row as `thrust_scale`. PID/fallback untouched.
8. **Cheap + synchronous.** One `rl_logger.log(row)` per **policy** step, plus one row
   on every `source` change (so faults are visible). All logging is wrapped in
   `try/except: pass` — logging never raises into the control loop.

### 3b. Adapter-reported `zero`/`geofence`

The brief's known DRL bug: `RLController.step` returns `np.zeros(3)` on a geofence
breach *before* `_emit`, bypassing the slew limit, and a non-finite session output
becomes `source="zero"`, `fault="nan"`. The mux therefore treats a `PolicyResult`
with `source in {"zero","geofence"}` as a **latched fault** (`source="zero"`,
`fault=` the adapter's fault or the source name) rather than letting a big un-slewed
step through. Not fixed in DRL, per the hard rules.

### 3c. `RL_MAX_THRUST_SCALE` default

Shipped default is `1.0` (`src/config.py`). The 0.5 case is exercised in the proof.

---

## 4. Verification actually run

Both scripts live under
`C:/Users/AORUS/AppData/Local/hermes/cache/scratch/` (**not** in the repo):
`t06_mux_proof.py` (fake adapter, fake navigator, fake clock) and
`t06_real_e2e.py` (real R7 `RLAdapter`). Run from the repo root:

```
cd E:/buoy-project/pi-control
.venv/Scripts/python.exe C:/Users/AORUS/AppData/Local/hermes/cache/scratch/t06_mux_proof.py
.venv/Scripts/python.exe C:/Users/AORUS/AppData/Local/hermes/cache/scratch/t06_real_e2e.py
```

### 4a. Fake-adapter assertions — full real output (`RESULT: ALL PASS`, exit 0)

```
adapter.step raised: boom                      <- expected: the "inference" row
Policy call jitter: gap 1000100.0 ms (target 100.0 ms, 1000000% off)   <- expected
====================
(1) PID default byte-identical to ThrustComputer.compute
  [PASS] PID default exact equal (surge)  1.0 vs 1.0
  [PASS] PID default exact equal (sway)
  [PASS] PID default exact equal (yaw)
  [PASS] selected == pid by default
  [PASS] source == pid
====================
(2) IDLE / MANUAL pass-through identical + never slew-limited
  [PASS] IDLE pass-through identical  0.0,0.0,0.0
  [PASS] IDLE not slew-limited (full step in 1 tick)
  [PASS] IDLE adapter never called
  [PASS] MANUAL pass-through identical  0.9,-0.8,0.7
  [PASS] MANUAL not slew-limited (full step in 1 tick)
  [PASS] MANUAL adapter never called
====================
(3) Fault ladder rows -> source+fault, finite action in [-1,1]
  [PASS] gps -> fallback/gps  fallback/gps
  [PASS] gps action finite in [-1,1]
  [PASS] imu -> fallback/imu  fallback/imu
  [PASS] mag -> fallback/mag  fallback/mag
  [PASS] watchdog -> fallback/watchdog  fallback/watchdog
  [PASS] watchdog action finite
  [PASS] fresh state clears watchdog  policy/None
  [PASS] step raises -> fallback/inference  fallback/inference
  [PASS] inference-fault action finite
  [PASS] nan -> fallback/nan  fallback/nan
  [PASS] slow -> fallback/slow_inference  fallback/slow_inference
  [PASS] slow_inference action finite
====================
(4) 3 consecutive faults latch to PID + fault='latched'
  [PASS] fault 1 not latched (source fallback, fault gps)  ('fallback', 'gps')
  [PASS] fault 2 not latched  ('fallback', 'gps')
  [PASS] 3rd consecutive fault -> latched  ('fallback', 'latched')
  [PASS] latched survives a good state
  [PASS] new command clears latch -> policy  policy/None
====================
(5) Geofence breach -> zero + nav.stop() + latched until new command
  [PASS] breach -> source=zero  zero
  [PASS] breach -> fault=geofence  geofence
  [PASS] breach -> navigator.stop() called  1
  [PASS] breach -> zero thrust
  [PASS] geofence latched until new command  fallback/latched
  [PASS] new command clears geofence latch  policy
====================
(6) unknown controller request -> 'pid' + warning
  [PASS] unknown request -> selected pid  pid
  [PASS] unknown request -> warning logged  ["Unknown controller request 'banana' -> 'pid'"]
====================
(7) Slew: no emitted step > rate_limit x dt except IDLE/MANUAL pass-through
  [PASS] policy slew respected (no step > 0.2)  max=0.2000
  [PASS] IDLE pass-through not limited
====================
(8) RL_MAX_THRUST_SCALE=0.5 halves the policy action, PID unchanged
  [PASS] scale applied then slew-limited on first tick  0.2
  [PASS] scaled policy reaches 0.5 (not 1.0)  0.5
  [PASS] scaled policy halved on interior
  [PASS] thrust_scale does not touch PID
====================
(9) Decimation: 20 ticks @ 50ms -> exactly 10 policy calls
  [PASS] exactly 10 policy calls in 20 ticks  calls=10
  [PASS] in-between ticks repeat the previous command  repeats=10
====================
RESULT: ALL PASS
```

Decimation detail: 20 ticks 50 ms apart, `policy_dt_s=0.1`, **exactly 10**
`adapter.step` calls; the 10 in-between ticks each repeat the previous command
verbatim. The scripted per-call actions differ, so a repeat is distinguishable from a
fresh step.

### 4b. Real R7 model end-to-end (real output)

Scripted 3-waypoint mission, 30 fake-clock ticks, `RL_ENABLED` forced on,
`RLAdapter('models/policy.onnx','models/policy_meta.json')`:

```
adapter.available = True reason = None
adapter.policy_dt_s = 0.1
selected (before first tick) = pid -> resolves per tick
source histogram (30 ticks):
   policy/None              30
final ThrustCommand: surge=1.000000 sway=0.909528 yaw=0.059819
final nav state: WAYPOINT_NAV wp_idx 0
MOCK only — synthetic motion, no vessel, no hardware.
```

**This is a mock, not a vessel.** The motion model is a crude 0.05 m/tick creep toward
the current waypoint, the clock is fake, and `FakeGPS` returns a constant COG. It
proves the mux drives the real ONNX session and emits sane commands (30/30
`policy/None`, finite, in range); it says **nothing** about how the policy behaves on
the water.

### 4c. Regression / integration checks

| command | result |
|---|---|
| `.venv/Scripts/python.exe -m py_compile src/control/controller_mux.py` | exit 0 |
| `.venv/Scripts/python.exe C:/.../scratch/t06_mux_proof.py` | exit 0, `RESULT: ALL PASS` |
| `.venv/Scripts/python.exe C:/.../scratch/t06_real_e2e.py` | exit 0, output in §4b |
| `.venv/Scripts/python.exe -c "from hardware_mock import install; install(); import src.control.controller_mux; print('IMPORT OK')"` | `IMPORT OK`, no warning (RL_ENABLED=False silent) |
| `.venv/Scripts/python.exe -m pytest test.py test_ml.py -q` | `50 passed` (baseline preserved) |

---

## 5. What was NOT done / open problems

1. **WS section 3c: `RL_MAX_THRUST_SCALE` default** — the config already ships `1.0`;
   I did not change config (not my file). The 0.5 behaviour is proven.
2. **Latency untested on real hardware** — `RL_MAX_INFERENCE_MS=50` against a laptop
   p50 of ~0.14 ms (T05 §4). The `slow_inference` branch is exercised only with a fake
   `inference_ms=999`. Must be re-measured on the Pi 3B before the 50 ms budget is
   trusted.
3. **`unavailable` latch path** — reachable only when the operator raised `RL_ENABLED`
   and requested `ppo` but the adapter is not available. The shipped
   `RL_ENABLED=False` never reaches it. Covered by code inspection; not separately
   scripted (the fake-adapter proof covers the other eight rows).
4. **Mock-vs-real clock coupling (found during this task, worth flagging to T07/T09)**
   — `RLAdapter` and `ControllerMux` each take a `clock`; they **must** be the same
   object. If the mux is given a fake clock while the adapter keeps the default
   `time.monotonic`, the adapter's internal watchdog compares the mux's `est["t"]`
   against real monotonic time and stamps every step `fault="watchdog"`. In production
   both default to `time.monotonic`, so this only bites tests / the mock — but any test
   that injects a clock into one side must inject it into the other.
5. **`RLController` resets its obs ring buffer whenever `source != "policy"`**
   (T05 §6). The first recovery step after a fault is therefore a degenerate 1-sample
   history. The mux does not compensate; treat post-fault steps as warm-up.
6. **`goal` for the policy** is built from `navigator.current_waypoint` (lat/lon, plus
   `heading_hold` when set). Station-keeping goals are **not** yet fed to the policy
   (`mode` there would need `STATION_KEEP` → `"station_keeping"` plus a hold point);
   only waypoint navigation drives the policy in this build. Flagged, not invented.
7. **T07 integration** — `compute(nav_output, state)` is drop-in for
   `self.thrust.compute(nav_output, state)`; `set_controller_request` is intended to be
   called from `_handle_remote_command`. Neither file is mine to edit.
