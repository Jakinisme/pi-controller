# Policy swap A@100M — 2026-11-07

The vendored RL policy in `models/` was replaced. Read this before trusting any number.

## What changed

| | before | after |
|---|---|---|
| source | `drl/runs/r7_policy.onnx` | `drl/runs/a1_long_A/policy.onnx` |
| ckpt | `r7_300m` (299,958,272 steps) | `a1_long_A` (99,942,400 steps) |
| reward | (R7 line) | A — the D3 winner |
| sha256 | `dbb90bb8...` | `90359115...` |
| onnx size | 356448 B | 356448 B (same architecture — size proves nothing) |

Applied with `tools/sync_rl_runtime.py --onnx runs/a1_long_A/policy.onnx --meta
runs/a1_long_A/policy_meta.json`; `--check` reports "in sync". `src/rl/*.py` were unchanged
(the DRL runtime files had not moved).

**R7 is not lost:** `models/backup_r7_20261107/` holds the previous `policy.onnx`,
`policy_meta.json` and `MANIFEST.json`. To roll back, copy those three back over `models/`.
(Restore the tests with `git revert 8f69911` — the old tests pin R7's behaviour.)

## Why the old tests went red

Three tests were written *against R7* and failed the moment the model changed. Two of the
three were failing because the new policy is **better**, which the old assertions read as a
regression:

| test (old name) | R7 measured | A@100M measured | verdict |
|---|---|---|---|
| `test_sway_does_not_distinguish_port_from_starboard` | split 0.003 | split **1.3394** | improvement, assertion inverted |
| `test_zero_error_at_goal_commands_a_stop_or_reverse` | surge +1.0000 | surge **+0.0782** | improvement, now passes |
| `test_action_parity_200_sequences` | compared to `runs/r7_policy.onnx` | — | test bug: hardcoded path |

The parity failure was **not** a real break: the test hardcoded `runs/r7_policy.onnx` as the
DRL-side model while the vendored model was A@100M, so it compared two different networks.
Fixed by reading the DRL-side path out of `models/MANIFEST.json`
(`test_rl_parity._manifest_model_paths`), which keeps the check working across future swaps.

## Measured behaviour of the new policy

`tools/describe_policy.py` prints this for whatever is installed. Current numbers:

`a_prev = 0` (episode first tick):

| goal | surge | sway | yaw |
|---|---|---|---|
| 50 m ahead | +0.8387 | -0.9551 | +0.6782 |
| 50 m astern | +1.0000 | +0.2569 | -0.9904 |
| 30 m starboard | +0.9636 | +0.3729 | +0.6472 |
| 30 m port | -0.9375 | -0.9665 | -0.9938 |
| 0 m (at goal) | +0.0782 | -0.0334 | -0.1360 |

`a_prev = A_PREV_ENV` (realistic mid-episode):

| goal | surge | sway | yaw |
|---|---|---|---|
| 50 m ahead | +0.9999 | -0.9915 | +0.5101 |
| 50 m astern | +1.0000 | -0.6264 | -0.9610 |
| at goal | +0.9895 | -0.9075 | -0.7203 |

Grid n=625: surge in [-0.9998, +1.0000], **frac(surge > 0.99) = 0.768**.

## The honest limit — do not skip this

**The policy is still largely `a_prev`-dominated, not goal-driven, in the regime a running
boat actually lives in.** With a realistic mid-episode `a_prev`, reversing the goal 180 deg
moves surge by **0.0001**, and at the goal it holds **+0.9895** surge (it keeps driving
forward). It reads as goal-responsiveness only on the episode's first tick, when `a_prev` is
zero. 76.8 % of the tested grid still sits above surge 0.99.

So: better than R7 on the two criteria R7 failed, but **not** a clean goal-tracker, and the
second assertion in `test_zero_error_at_goal_commands_a_stop_or_reverse` pins the bad
mid-episode number on purpose so it cannot be forgotten.

## Verification run

`pytest -q` in `pi-control` -> **125 passed** (58 warnings, all from unrelated sklearn ML
artefacts), 7.26 s. `sync_rl_runtime.py --check` -> "in sync".

## Still not a real-vessel claim

Everything above is simulator/ONNX behaviour. `drl/SYSID.md` status is "nothing measured" —
mass, inertia, added mass, drag and thrust are still placeholders, INSPECTION H6 (sway/yaw
sign) is unresolved, and the ESP32 failsafe is 2 s (H5). The measured `frac(surge>0.99)=0.768`
is a saturation warning that matters a great deal on real hardware (current draw, heat,
battery) and not at all in the sim.
