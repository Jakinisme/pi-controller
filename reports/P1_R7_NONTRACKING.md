# P1 — Why R7 does not track in the mock (investigation, 10 sub-agents)

Scope: diagnosis only. Nothing here was fixed, nothing committed, no hardware run.
Every number comes from executed code on this laptop; none of it says anything about
the real vessel. Sources are listed per finding.

---

## 0. TL;DR

Three separate things were conflated. All three are now resolved:

| claim | verdict | evidence |
|---|---|---|
| Pi integration (mux / adapter / vendored runtime) is wrong | **REFUTED** | `reports/P1_T09_vendor_audit.md`: bit-identical, 0.0 max diff |
| The R7 policy cannot track anything | **REFUTED** | real simulator: 3.88/4.00/3.50 of 4 waypoints, 96.9/100/87.5 % lap |
| The mock's stand-in vessel is the cause | **PARTLY REFUTED** | sign bug real, fixing it moves 17.88 → 15.74 m only |

What is actually wrong:

1. **R7 is a bang-bang pass-through controller.** It reaches waypoints in its own
   simulator but never settles: `settling_time_s = inf`, `settled_pct` 0–37 %.
   It flies through the 2 m arrival radius at speed.
2. **The mock's stand-in vessel has an inverted sway sign** (`tools/mock_smoke.py:93`),
   plus a plant that is not representative (instant yaw, no inertia, `MAX_SPEED` 1.5 vs
   the sim's `max_surge_mps` 1.0, 20 m mock geofence vs 80 m in training).
3. **A latent clock-domain bug in the mux/adapter contract** makes the policy emit zero
   thrust forever whenever a caller injects its own `now`. This one is real, reproduces,
   and should be fixed before any tethered trial.

---

## 1. A correction I have to state first

An earlier reading from this investigation — mine, then reproduced by two sub-agents —
said "R7 emits surge ≈ +1.0 for every input, it is not a controller at all".

**That was a measurement artefact.** The static feed used `a_prev = (0,0,0)`, `mode = 0`
and only positive `goal_dx` — a regime the simulator visits only on the very first tick.
Measured on the real ONNX with the real `raw_sample`, holding everything else fixed:

| `a_prev_surge` | goal 20 m ahead | goal 0 m | goal 20 m astern |
|---|---|---|---|
| `0.000` | `+0.9998` | `+1.0000` | `+0.9999` |
| `-0.992` (env mid-episode value) | `+0.9999` | `-0.6191` | `-1.0000` |

The previous-action channels (7–9, carried K=10 deep in the history) are what move
surge, and the simulator's own `results_r7/summary.json` and a real CUDA replay
(agent 5) both show `surge` spanning `[-1.000, +1.000]`, `mean +0.473`, 63 % of steps
above +0.9 and 22 % below −0.9. The policy reacts; it simply reacts badly.

Anyone reading `tests/test_rl_action_plausibility.py` should read it as *"the action
table is wrong for a well-behaved controller **under this feed**"*, not as proof of a
dead network.

---

## 2. What R7 actually does in its own simulator

Real run, `drl/.venv` (torch 2.6.0+cu124, `cuda True`), the same checkpoint the export
came from (`steps 299958272`, `iteration 4577`, `obs_mean|max| = 0`, `obs_std = 1`):

| run | waypoints_reached | lap_pct | lap_completed | radial_error_mean |
|---|---|---|---|---|
| recorded `results_r7` (ppo, 4 seeds) | 3.78 | 94.5 % | 0.844 | 13.0 |
| re-run `--seeds 4`, wp_square | 3.88 / 4.00 / 3.50 | 96.9 / 100 / 87.5 | — | 12.67 / 13.16 / 14.30 |
| PID control, same seeds | 0.00–0.38 | 0–9.4 % | 0 | 26–34 |

R7 is roughly 10× better than the PID baseline in its training environment. But:

- `settling_time_s = inf` on every run.
- `settled_pct` = 0–37 % at `tol = 1.0 m`.
- `radial_error_max ≈ 32 m` on a 30 m square.

So the waypoint counter is satisfied by driving *through* the 2 m radius, not by
arriving and holding. R7 is a pass-through waypoint runner, not a station-keeper. That
is the single most important behavioural fact for deployment, and it was never
regression-gated: no test in `drl/tests/` asserts that a trained policy produces a
sensible action — `test_onnx_parity.py` and `test_runtime.py` check shapes, parity and
`|a| ≤ 1` only, and every runtime test uses a `dummy_onnx` random MLP, never the real
`runs/r7_policy.onnx`.

Training geometry note: `evaluate.py` used `square_side = 30.0`. The mock's square has
50 m legs reaching 70.7 m. R7 was never evaluated beyond ~30 m legs.

---

## 3. What actually happened in the mock (recorded data)

`reports/P1_trace_analysis.md`, from `logs/rl_smoke/rl_20261028_233320.csv` (145 rows,
t = 0 → 14.35 s, goal 50.038 m due north, `wp_idx = 0` the whole run):

- Range to goal decreased on **286/286 comparable policy steps**. The vessel is not
  diverging and the command mapping has the right sign.
- Final position `east +9.27 m`, along-track only 17.68 m of the 50 m leg (35.3 %).
- Heading error grew `-15.1°` → `-50.2°` and never recovered.
- `surge` mean `+0.9885`, pinned at the rail on 96.6 % of rows. `sway` mean `+0.8534`.
  `yaw` mean `+0.1186`, **not** saturated, and `corr(sway, yaw) = -0.706`.
- Latch: 19.964 m from the first fix → the mux's own `RL_GEOFENCE_M = 20.0` fires,
  `source = zero`, `fault = geofence`, `navigator.stop()` → IDLE. That part is the
  safety layer working exactly as designed.

Net: the mock flight is a wide diagonal that eats the 20 m fence before the 50 m leg
closes. Not a spin, not a divergence.

---

## 4. The mock stand-in vessel (`tools/mock_smoke.py`)

### 4.1 Confirmed bug — inverted sway

`SimpleVessel.step` has `v_t = -cmd.sway * MAX_SPEED_MPS * SWAY_GAIN` (line 93). Measured
with a pure sway command:

| heading | `sway = +1` result | project convention (`thrust.py:45`, `frames.body_to_enu`) | verdict |
|---|---|---|---|
| 0° (bow north) | east **−2.70 m** | sway + = starboard = +east | **INVERTED** |
| 90° (bow east) | north **+2.70 m** | sway + = starboard = −north | **INVERTED** |

The body→ENU rotation in the same function is correct (`ve = u·sinψ + v·cosψ`,
`vn = u·cosψ − v·sinψ` matches `common/frames.body_to_enu`). The negation alone flips the
axis.

Locked by `tests/test_rl_mock_model.py`: **6 failed, 9 passed** on the current repo — all
6 failures are this sign, nothing else. Against a sign-corrected model the file is
15 passed.

### 4.2 It is not the cause of the non-tracking

`tools/mock_push_test.py`, identical scripted input, both plants:

| | shipped (sway+ → port) | corrected (sway+ → starboard) |
|---|---|---|
| final range | 17.88 m | 15.74 m |
| east offset | +5.42 m | +3.91 m |
| final heading | ~102° | ~97.5° |

Removing only the negation makes the two plants agree to 0.05 m. The sign matters for
correctness, not for this failure.

### 4.3 Other ways the stand-in differs from where R7 was trained

`MAX_SPEED_MPS = 1.5` vs the sim's `max_surge_mps = 1.0`; `YAW_RATE_DEG_S = 20` applied
instantly with no inertia vs the sim's `inertia_izz_kgm2 = 1.2`, added mass, drag and
thruster lag `tau = 0.12 s`; `SWAY_GAIN = 0.2`; and `course_deg` is stamped equal to
`heading_deg` in `to_state()`, so the SOG/COG body-velocity estimator always sees
`dg = 0` and the policy never observes lateral drift it is itself causing.

Practical consequence: the mock lines up with a policy that wants to hold full thrust
and does not model the yaw inertia R7 was trained around. It is a wiring harness, not a
plant model, and should stop being used as evidence about tracking.

---

## 5. New bug found — clock domain mismatch (REAL, not a mock artefact)

`src/control/controller_mux.py:517` puts the mux's own `now` into `estimator_state["t"]`,
while `RLController._check` (`src/rl/policy.py:164`) compares that value against its own
`self._clock()`, which the adapter wires to `time.monotonic()` (`src/rl/adapter.py:75,128`).
Any caller whose `now` is not in the monotonic domain — an injected clock, a replay
driver, a test, `runtime/infer_replay.py` — is permanently older than the watchdog.

Reproduced:

```
A) "t" = time.monotonic()      step0..3 src=policy   fault=None
B) "t" from a virtual clock    step0..3 src=fallback fault=watchdog  action=(0,0,0)
```

Case B emits **zero thrust forever**. `tools/mock_smoke.py` passes a real monotonic clock
so it never trips, which is why this was invisible. `src/rl/policy.py` is vendored and
read-only; the fix belongs on the Pi side (either pass `now` in the controller's clock
domain, or stop writing `t` from a foreign clock). Not fixed in this investigation.

---

## 6. Items considered and cleared

- **Vendored runtime drift** — `reports/P1_T09_vendor_audit.md`: only 4 import lines
  differ (AST-proven), everything else bit-identical at 0.0 max abs diff over ~300 k
  values, including wrap boundaries, the SOG gate and the K=10 ring order. Manifest
  sha256 for all 6 files verifies; `--check` catches a blessed-mutation attempt.
- **ONNX export** — checkpoint weights vs `models/policy.onnx` agree to `1.78e-06`
  (hand-rolled numpy MLP, no onnxruntime). `obs_mean = 0`, `obs_std = 1`, so the graph's
  baked normalisation is a no-op and there is no double normalisation.
- **Wrong model vendored** — `sha256(pi-control/models/policy.onnx)` =
  `sha256(drl/runs/r7_policy.onnx)` = `dbb90bb8…`. The older non-R7 export is
  `8dd323b3…`; same byte size, different file.
- **ckpt vs recorded results** — `results_r7/summary.json` names
  `runs/r7_300m/ckpt/ckpt_last.pt`, `steps 299958272`, `iter 4577`; the eval ran 16:13–16:39
  after the final training step at 15:57. The summary is not stale.
- **`fast_env_parity.obs = 4.838`** — a red herring. `evaluate.py::check_fast_env`
  documents that obs/state differences are by design (`sim/sensors.py` draws noise from
  the global torch RNG); only `dist` is an equality claim, and it is 0.0.
- **`log_std = [-0.1347, -0.9345, -1.1405]`** → action std `[0.874, 0.393, 0.320]`. Still
  unexplained. High for a finished policy and consistent with bang-bang behaviour.

---

## 7. Fixes applied

Both bugs from §4.1 and §5 are now fixed, with regression tests that were proven able to
fail (negative control: reverting the fix turns the test red).

| # | file | change |
|---|---|---|
| 1 | `tools/mock_smoke.py` | `v_t` was `-cmd.sway * …`; now `cmd.sway * …`. The body→ENU rotation was already correct, so the negation alone inverted the lateral axis. |
| 2 | `src/control/controller_mux.py` | `_build_est` stamped `"t": now` (the mux's own clock). It now uses `_policy_clock()`, which reads the adapter's clock so the field is in the same domain `RLController._check` compares against. Falls back to `now` if the clock is unreadable. |

New tests:

- `tests/test_rl_mock_model.py` — 15 tests pinning the stand-in body→ENU mapping to
  `frames.body_to_enu` at 0/90/180/270°, both axes. Was **6 failed / 9 passed** before
  fix 1; now **15 passed**.
- `tests/test_rl_clock_domain.py` — 2 new tests (a virtual clock must still yield
  `source="policy"`, and `est["t"]` must come from the adapter clock). Both pass after
  fix 2; the second one fails when the fix is reverted.

`tests/test_rl_action_plausibility.py` was **corrected**, not silently weakened: its
feed used `a_prev = 0`, the regime that made everything look saturated (§1). With a
realistic `A_PREV_ENV = (-0.992, -0.950, -0.738)` the measured behaviour is different and
the tests now lock what is actually true:

| property | measured | verdict |
|---|---|---|
| surge responds to `a_prev` | `+0.9999` (a_prev=0) vs `-0.6191` (a_prev=env) at goal | reacts |
| surge responds to goal direction | ahead `-0.9936` vs astern `-0.9995` — 0.006 for a 180° reversal | **does not track goal** |
| sway distinguishes port/starboard | `-0.9953` vs `-0.9986` — 0.003 for a 60 m swing | **blind to goal side** |
| at goal, zero error | surge `-0.6191` | backs off |

Full suite: **120 passed** (was 94 passed before this investigation, 9 failed mid-way).

The mock smoke still reports `0/4` waypoints with a geofence latch at 14.5 s. That is
expected and unchanged: fixing the sway sign and the clock does not make R7 track in a
plant model it was not trained on. It is recorded, not papered over.

---

## 8. What remains open

1. **Do not use the mock to judge tracking.** If a plant model is wanted, port
   `sim/dynamics.py` + `sim/thrusters.py` + `sim/sensors.py` (~200 MB CPU-only torch; one
   env on CPU is comparable to CUDA because `AsvEnv.step` already does 12 `.item()`
   syncs; ~3–5 min per 200 s episode). Reasonable offline, not on the Pi at runtime.
2. **Treat R7 as a pass-through waypoint runner, not a station-keeper.** `settled_pct` 0–37 %
   and `settling_time_s = inf` mean it will not hold a point. For station keeping on
   water the PID path is the honest choice until R7 is retrained or wrapped with a hold
   controller.
3. **R7's sway is not a lateral-correcting channel** (measured above: blind to goal side).
   Any steering authority must come from yaw, and yaw was already observed near zero under
   a 50° heading error in the real mock CSV (§3).
4. **Add a behaviour gate to the DRL suite.** No test ever asserted that a trained policy
   produces a sensible action; that is how a pass-through policy shipped as "R7 works".

Nothing above is evidence about the real vessel. `H6` (sway/yaw physical sense) remains
open and still requires the props-off bench test before any water trial.
