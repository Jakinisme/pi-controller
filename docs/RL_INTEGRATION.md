# RL_INTEGRATION — running the PPO (R7) policy on the ASV

The operating manual for the on-board PPO controller. **The policy is DISABLED by
default** (`RL_ENABLED = False`, controller `"pid"`). Nothing in this document has
been run on the vessel or the Pi.

Read this with `reports/P1.md` (design + findings) and `reports/P1_smoke.md`
(mock run). Nothing in the simulator is evidence for the real vessel.

| item | value |
|---|---|
| model | R7 — `models/policy.onnx` (sha256 in `models/MANIFEST.json`) |
| runtime | onnxruntime + numpy only. **No torch on the Pi.** |
| policy rate | `policy_dt_s = 0.1` (10 Hz), 10-step observation history |
| control rate | `FUSION_UPDATE_RATE_HZ = 20` (20 Hz); the mux decimates the policy to 10 Hz |
| default | PID. The policy only runs if asked for AND enabled |

---

## 1. Exporting a new policy from DRL and syncing it

In the DRL repo (`E:/buoy-project/drl`, read-only from here):

```
python export_onnx.py --ckpt runs/<name>/ckpt/ckpt_last.pt \
                      --out  runs/<name>_policy.onnx \
                      --meta runs/<name>_policy_meta.json
```

Defaults: `--ckpt runs/checkpoints/ckpt_last.pt`, `--out runs/policy.onnx`,
`--meta` beside `--out`. It writes an ONNX graph `obs float32 [B,150] ->
action float32 [B,3]` (deterministic tanh mean, opset 17) plus the meta json.

Then vendor it into this repo:

```
cd E:/buoy-project/pi-control
.venv/Scripts/python.exe tools/sync_rl_runtime.py \
    --drl E:/buoy-project/drl \
    --onnx runs/r7_policy.onnx \
    --meta runs/r7_policy_meta.json
```

This copies the four vendored runtime files (`src/rl/{policy,obs_np,estimator,frames}.py`,
with the documented import rewrites and nothing else changed), the ONNX model and
its meta into `models/`, and rewrites `models/MANIFEST.json`. It prints one line
per file (`created`/`unchanged`/`changed`) and is **idempotent** — run it twice and
the second run reports everything unchanged; `synced_utc` is only re-stamped when
something actually changed.

Verify without writing anything:

```
.venv/Scripts/python.exe tools/sync_rl_runtime.py --check
```

`--check` exits **0** when the vendored copy + manifest match DRL, and exits **1**
with a diff list otherwise. Run it after any DRL pull and before every deployment.

**What `models/MANIFEST.json` guarantees.** For every vendored file it records
`sha256`, `bytes` and the DRL-relative `src`, plus `drl_commit`, `drl_path`,
`checkpoint_steps`, `iteration` and `source_ckpt`. So the manifest is the
answer to "which checkpoint, from which DRL commit, is actually on the boat" —
compare the shipped `policy.onnx` sha256 against it, and compare `drl_commit`
against `git -C <drl> rev-parse HEAD`. If they differ, the runtime was not synced
after the last DRL change.

Current manifest: `drl_commit 608256b224dc5e81dc541175df94256399fba4d4`,
`checkpoint_steps 299958272`, `iteration 4577`,
`source_ckpt runs/r7_300m/ckpt/ckpt_last.pt`; `policy.onnx` sha256
`dbb90bb8b0123dc9276f0fefc3ff270937992a8fa30e8a14e617a8af41c49913`.

> Never use `runs/policy.onnx` from DRL — it is R6-era. R7 is the newest.

## 2. Installing on the Raspberry Pi 3B

Assumptions: **Raspberry Pi OS Lite 64-bit (bookworm)** on a Pi 3B (Cortex-A53,
64-bit capable). A 32-bit (armv7) image has **no** onnxruntime wheel at all — do
not use one.

```
sudo apt update && sudo apt install -y python3-venv
cd ~/pi-controller
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins `onnxruntime==1.31.1`. Its aarch64 wheel is
`manylinux_2_28_aarch64` (glibc ≥ 2.28); Bookworm ships **glibc 2.36**, so it
installs with margin. Do not pin the older `manylinux2014_aarch64` tag — pip
resolves nothing newer than `onnxruntime<=1.16.3` for it.

**Measure inference on the Pi itself** before trusting `RL_MAX_INFERENCE_MS = 50`.
The wheels and the laptop numbers are not the Pi numbers:

```
cd ~/pi-controller
.venv/bin/python tools/bench_inference.py --n 1000
```

Exact command, from `reports/P1_T05_adapter.md` §4 (that output was produced on an
x86 laptop — the Pi's will be slower):

```
.venv/bin/python tools/bench_inference.py --n 1000
note: RL_MODEL_PATH='C:\Users\AORUS\pi-controller\models\policy.onnx' absent, benching E:\buoy-project\pi-control\models\policy.onnx
onnxruntime=1.31.1  policy_dt_s=0.1  n=1000
inference ms: p50=0.143  p95=0.244  max=0.465  mean=0.170  (n=1000, warmup=20)
```

(On the Pi the configured `RL_MODEL_PATH` exists, so the note does not appear.)
Also watch RSS and load while the full system runs (`vmstat`/`ps`) — Pi 3B CPU/RAM
headroom with sensors + Firebase + onnxruntime alive at once is **unmeasured**.

## 3. Enabling and rolling back

**Enable** — two things:

1. `src/config.py`: `RL_ENABLED = True`.
2. Ask for it. Either `RL_DEFAULT_CONTROLLER = "ppo"`, or leave it `"pid"` and set
   the `controller` field in a Firebase command (`"ppo"`).

```
{ "mode": "waypoint", "controller": "ppo", "waypoints": [...] }
```

`controller` is optional; missing → `"pid"`; anything other than `"pid"`/`"ppo"`
→ `"pid"` with one warning. It is parsed **inside** the command and is subject to
the existing timestamp gate, so a `controller` field left in `commands/` from
before a restart is ignored and the system comes up IDLE/PID.

Telemetry gains a `control` node under `/vehicles/{id}/telemetry/` carrying
`controller` (the selected one), `source` and `fault` — see §7.

**Roll back** — set `RL_ENABLED = False` and restart. That is the whole procedure:
the PID path is unchanged and is the default, the mux passes `IDLE`/`MANUAL`
straight through to `ThrustComputer` byte-identically, and with `RL_ENABLED=False`
the adapter is not even built. There is no state to unwind. You can also roll back
without a restart by sending a command with `"controller": "pid"`.

## 4. Pre-water checklist — do these IN ORDER

Each step is a gate. Do not start the next until the previous one passes.
Props off until step 4 says otherwise.

| # | step | pass evidence |
|---|---|---|
| 1 | **Props-off bench test of the sign table** against the firmware `MIX_MATRIX` in `firmware-esp/src/header/config.h` (rows NW/NE/SW/SE; `motor_invert=[false,true,false,false]`). Command a known `[surge, sway, yaw]` with props removed and check every motor spins in the direction the mixer says. | every one of the 4 motors matches the MIX_MATRIX row for its corner; any disagreement is a STOP, not a wiring fix |
| 2 | **Manual per-axis motor test from the dashboard.** One axis at a time, small values, `mode:"manual"`. | each axis moves the hull the way `+surge`/`+sway`/`+yaw` claims (INSPECTION H6 is settled here or the run is cancelled) |
| 3 | **Mock-hardware run.** `tools/mock_smoke.py --seconds 60` on the Pi, Firebase off. | exit 0, "every emitted command finite and within [-1,1]", and the `inference_ms` p95 from the Pi is well under `RL_MAX_INFERENCE_MS` |
| 4 | **Tethered run with a small geofence.** `RL_GEOFENCE_M` small (a few metres), `RL_MAX_THRUST_SCALE` low, a short one- or two-waypoint mission, vessel tethered, operator on the kill switch. | the vessel moves toward the waypoint, the mux trips the geofence where expected, and a trip gives zero thrust + IDLE + latch — never a runaway |
| 5 | **RC / manual kill path verified.** Prove the manual `idle`/`manual` path and the physical kill both work while the policy is flying. | a manual command stops the policy within one tick; the kill switch stops the motors |
| 6 | **Calm-water station keeping** before any waypoint run. `mode:"station_keep"`. | the vessel holds position/heading in calm water with the policy, and you have seen the last-waypoint behaviour below |

## 5. What the first tethered trials must record, and why

Record the full `RLLogger` CSV (`logs/rl/rl_YYYYmmdd_HHMMSS.csv`) — it is the only
record of commanded vs. achieved behaviour.

**Commanded vs. achieved speed.** The CSV carries the applied action
(`action_applied_surge/sway/yaw`) and the measured `speed_mps`. Log both, on the
same clock. The vessel currently **creeps and cannot push against wind, current or
waves** — and the simulator's thrust model does **not** reproduce that. A policy
trained with generous simulated thrust may command a surge the real hull simply
cannot deliver, so until you have compared commanded against achieved you cannot
tell a control fault from a thrust shortfall.

**Why the CSV is the sysid input.** The same file is the input to system
identification: it has `t_monotonic` (seconds since the first row — usable as a
replay clock), lat/lon, heading, speed and the applied action. From it you run a
**bollard pull** (command a known surge, vessel held, record the force/steady
speed) and **decay tests** (spin up, cut thrust, watch the speed decay) to replace
the placeholder physics with measured coefficients. That is what turns the policy
from "trained in a made-up boat" into something with a measured platform under it.

Note the logger/replay gap: `runtime/infer_replay.py` can replay the required and
most optional columns, but `goal_heading_deg`, `next_lat/lon`, `next2_lat/lon` and
`est_age_s` are **not** in the header, so station-keeping heading and the
later-waypoint observations cannot be replayed from the log.

## 6. Known limits — plainly

- **Placeholder physics** in the training simulator, and **INSPECTION H6 is still
  open** (the physical sense of sway/yaw on the real hull is unverified until the
  props-off bench test).
- **`u`, `v` are derived from SOG/COG — there is no body-velocity sensor (H7).**
  Above 0.3 m/s this is a coarse estimate; below it, the policy decays the last
  estimate. The policy has never seen real GPS lag (~1 Hz NMEA).
- **`next`/`next2` waypoints are absent in the goal the mux builds**, so the
  look-ahead the policy trained with is missing (observed independently in the
  mock smoke: `src/control/controller_mux.py::_build_goal` sets only
  `lat`/`lon`/`mode`/`heading_deg`).
- **At the LAST waypoint the navigator goes IDLE, not to a hold point.** Training
  treats the final waypoint as a station-keeping hold (target velocity 0); on the
  Pi `WaypointNavigator` calls `stop()`, so the vessel goes idle and drifts. Not a
  safety bug, but the end-of-mission behaviour seen in the simulator is **not**
  reproduced.
- **Heading is a trusted input.** `HEADING_MOUNT_OFFSET_DEG` and
  `MAGNETIC_DECLINATION_DEG` are calibration constants the policy simply believes.
  Get them wrong and the policy is confidently wrong.
- **The policy was never trained on real sensor latency**, the Pi's ~1 Hz GPS, or
  the 20 Hz loop's jitter. The mux logs a rate-limited jitter warning so this is
  visible in the field.
- **The mux geofence (20 m, `RL_GEOFENCE_M`) is a deployment knob, not the 80 m the
  policy trained with** (`models/policy_meta.json` `safety.geofence_m = 80.0`). The
  mux's tighter one runs first and latches.
- **`RL_MAX_THRUST_SCALE` below 1.0 changes behaviour** — the policy never trained
  with its actions scaled down. Use it only as a deliberate trial restriction.
- **Pi 3B CPU/RAM headroom is unmeasured.** `RL_MAX_INFERENCE_MS = 50` is an
  unvalidated deployment threshold until §2 is run on the Pi.
- **Vendored-code quirk (not fixed; DRL is read-only).** `RLController.step` returns
  zeros *before* its slew limiter on a geofence breach, and a non-finite output
  surfaces as `source="zero"`, `fault="nan"`. The mux therefore owns the tighter
  geofence, its own latch, and treats `source in {"zero","geofence"}` as a fault.

## 7. The Firebase `controller` field (for the dashboard team)

The React project is **not** changed here. This is only the on-board contract.

- **Command field.** Add an optional `controller` string to any command node under
  `commands/`: `"pid"` or `"ppo"`. Missing → `"pid"`. Any other value → `"pid"` and
  the Pi logs one warning. It is parsed inside the command and passes the existing
  timestamp gate, so it cannot be applied from a pre-restart node.
- **Telemetry.** Read `controller`, `source` and `fault` from a new additive
  `control` sub-node: `/vehicles/{id}/telemetry/control/`. `controller` is the
  selected controller (`pid`/`ppo`), `source` is the last emitted source
  (`policy`/`fallback`/`geofence`/`zero`), `fault` is the last fault string (empty
  string when none — Firebase rejects `null`). A new node was chosen over reusing
  `nav` so no existing node's shape changes.

## 8. What NOT to claim

- **No simulator result is evidence for the real vessel.** Not the R7 evaluation
  numbers, not the mock smoke, not any dry-run. They exercise code paths; they say
  nothing about how the hull behaves in water.
- The mock smoke (`reports/P1_smoke.md`) has no vessel, no hardware, no GPS and no
  network, and its timings are a laptop's, not the Pi 3B's.
- The policy is **off** unless someone turns it on. The default is PID.
