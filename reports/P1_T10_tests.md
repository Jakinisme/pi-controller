# P1 / T10 — fault matrix, decimation and logger tests

Owner of exactly: `tests/test_rl_faults.py`, `tests/test_rl_decimation.py`,
`tests/test_rl_logger.py`, this report. No product code touched. Tree left dirty
for the orchestrator to commit.

Environment: `E:/buoy-project/pi-control`, runner `.venv/Scripts/python.exe`
(Python 3.13.3). Branch `feat/ppo-controller`, checkpoint `446e0f8`.
Fixtures reused verbatim from `tests/conftest.py` + `tests/_fakes.py`
(`FakeAdapter`, `ScriptedNavigator`, `FakeGPS`, `make_state`, `make_nav`);
nothing duplicated.

## Runner output (exact)

```
cd E:/buoy-project/pi-control && .venv/Scripts/python.exe -m pytest -q
........................................................................ [ 76%]
.........x............                                                   [100%]
93 passed, 1 xfailed in 5.62s
```

* 94 collected total = **64 pre-existing** (`test.py`, `test_ml.py`, the four
  T09 files — re-run alone: `64 passed`) + **30 new** from this task.
* New files alone: `29 passed, 1 xfailed in 1.34s`.
* The one `xfailed` is a real product bug, documented below.

## 1. `tests/test_rl_faults.py` — the full ladder (13 tests)

One test per ladder row; each asserts the exact `(source, fault)` pair read from
the real attribute names (`mux.source`, `mux.fault`, `mux.selected`,
`mux._latched`), a finite command on all three axes, and (for the slew test)
that no emitted command steps by more than `rate_limit_per_s * policy_dt_s`
(2.0 × 0.1 = 0.2) — except the `IDLE`/`MANUAL` pass-through.

| row | test | asserted |
|---|---|---|
| `gps_valid` False | `test_sensor_invalid_rows[gps_valid-gps]` | `fallback`/`gps`, 0 policy calls |
| `imu_valid` False | `test_sensor_invalid_rows[imu_valid-imu]` | `fallback`/`imu`, 0 policy calls |
| `mag_valid` False | `test_sensor_invalid_rows[mag_valid-mag]` | `fallback`/`mag`, 0 policy calls |
| stale sample | `test_stale_sensor_watchdog` | `fallback`/`watchdog`; a *changed* sample clears it |
| `adapter.step` raises | `test_adapter_step_raises_inference` | `fallback`/`inference` |
| non-finite action | `test_non_finite_action_nan` | `fallback`/`nan` |
| `inference_ms` > budget | `test_slow_inference_over_budget` | `fallback`/`slow_inference` |
| 3 consecutive faults | `test_three_consecutive_faults_latch_until_new_command` | 3rd → `fallback`/`latched`, survives a good state, cleared by `set_controller_request` **and** by `reset()` |
| geofence breach | `test_geofence_breach_zero_plus_stop_plus_latch` | `zero`/`geofence`, exactly-zero thrust, `navigator.stop()` called once, `navigator.state==IDLE`, latched until new command |
| adapter unavailable while ppo requested | `test_adapter_unavailable_while_ppo_requested_latches` | `fallback`/`unavailable` + latched; next tick `latched`; cleared then re-fires |
| **`no_goal`** (orchestrator) | `test_no_goal_autonomous_with_empty_navigator` | `fallback`/`no_goal`, command is a **real PID output (not zero, not NaN)**, 0 policy calls |
| `no_goal` × 3 | `test_no_goal_does_not_run_but_three_consecutive_latch` | latches like every other ladder fault |
| slew + pass-through | `test_no_emitted_step_exceeds_rate_limit_except_idle_manual` | max step ≤ 0.2 over a policy run + a fault + recovery; `MANUAL` byte-exact vs `ThrustComputer.compute` |
| ladder order | `test_ladder_order_gps_outranks_imu_mag_and_geofence` | all-bad state → `gps`; then `imu`; then `mag` (first match wins, sensor beats geofence) |

## 2. `tests/test_rl_decimation.py` — decimation & jitter (6 tests)

* **`test_twenty_ticks_fifty_ms_yield_exactly_ten_steps`** — 20 ticks at 50 ms
  with an injected fake clock → `len(adapter.calls) == 10` exactly (asserted, not
  inferred), `selected == "ppo"`, `source == "policy"`.
* **`test_in_between_ticks_repeat_previous_command_verbatim`** — the 10 odd ticks
  each equal the preceding command on all three axes; the scripted per-call
  actions differ by 0.05 so a repeat is distinguishable from a fresh step
  (guarded by its own assert).
* **`test_jitter_warning_fires_over_thirty_percent`** — a 200 ms gap (100 % off)
  logs **exactly one** `controller_mux` WARNING whose message contains
  `"200.0 ms"`.
* **`test_no_jitter_warning_within_thirty_percent`** — a 110 ms gap (10 % off)
  logs **zero** warnings.
* **`test_jitter_warning_rate_limited_over_a_burst`** — 20 real `compute` calls
  at 200 ms each (4.0 s of fake time) → **exactly 1** warning (the 5 s limiter);
  a further 8 s burst warns 1–2 times. Call count via `caplog`, driven through
  the public `compute` only.
* **`test_first_tick_always_steps_regardless_of_jitter`** / **`test_repeat_does_not_call_step_and_keeps_finite_range`** — first-tick step, and a 50 ms
  repeat makes no extra `step` call and stays within `[-1, 1]`.

## 3. `tests/test_rl_logger.py` — 25 columns, row count, slow disk, rotation (10 tests)

* **Header/row count** — `test_mixed_run_full_header_and_row_count_rule`: a
  mixed policy→fault→policy run through the real mux; every one of the 25
  columns present on every row, and the count equals the asserted rule:

  > **rows = policy_steps + fallback_entries**, where `policy_steps` is
  > `len(adapter.step)` calls (each writes exactly one row via `_emit_log_row`,
  > which also sets `_last_source_logged = "policy"`) and `fallback_entries`
  > counts one row each time the mux enters a non-policy source from a
  > *different* logged source. A return to `policy` writes **no** extra row —
  > the next policy step's own row carries `source="policy"`.

  The final logged sequence is asserted literally:
  `["policy","policy","policy","policy","fallback","policy","policy"]`.
* **`test_row_count_rule_is_policy_steps_plus_source_changes`** — pins the rule
  deterministically: 3 policy steps + 3 gps-fault ticks → exactly **4** rows
  (`policy, policy, policy, fallback`/`gps`); the 2nd/3rd fault ticks are
  neither a step nor a source change and log nothing.
* **goal columns** — `test_goal_lat_lon_populated_with_waypoint` (two waypoints
  → both `(lat, lon)` pairs appear in the goal columns); `test_goal_lat_lon_blank_without_waypoint` (no waypoint → goal columns are **blank strings**,
  not `0`).
* **Slow disk** — `test_slow_disk_never_delays_a_control_tick`: `_format` is
  monkeypatched to `sleep(0.05)`; over 20 mux ticks **nothing hits the slow path**
  (`log()` is a pure enqueue), total `compute` time across 20 ticks stays under
  5 ms, a single `log()` call stays under 5 ms, and after `close()` all 21 rows
  are on disk.
* **Rotation** — `test_rotation_with_small_max_bytes_loses_no_row`: `max_bytes=200`,
  6 rows → `_part2.csv` and `_part3.csv` exist, each part carries the full
  25-column header, and the parts contain **exactly 6** data rows (no loss).
* **`t_monotonic` relative + replay columns** —
  `test_t_monotonic_relative_and_replay_columns_present`: an epoch-scale fake
  clock (12345.0) still yields first row `0.000000` and successive rows at
  0.1 s steps, all `< 1.0` (proves it is *relative*, not the raw monotonic
  value); the replay-required columns `t_monotonic, lat, lon, heading_deg,
  speed_mps` are all present, and the header equals the frozen 25 exactly.

## 4. Product bug found (reported, not fixed — no product code touched)

**`RLLogger` loses every row queued after an unformattable one.**

`RLLogger._format` raises when a row carries a non-numeric `t_monotonic`
(`_t_rel` does `float(t) - float(self._t0)` with no guard). The exception
propagates out of `_write_all`'s `while self._queue:` loop, so the rows *after*
the bad one are never written; `self._errors` increments once and nothing is
raised into the caller (the control loop is safe, but the trace silently loses
rows).

Reproduced:

```python
lg = RLLogger(tmpdir, flush_interval_s=0.0)
lg.log({"t_monotonic": "not-a-number"})   # unformattable
lg.log({"t_monotonic": 1.0, "lat": 5.0})  # good row, queued after it
lg.close()
# lg._errors == 1, and the CSV holds 1 data row — the good row is GONE
```

Suggested fix (T08's file, not mine): make `_format` row-local — wrap the
`self._format(row)` call in `_write_all` in its own `try/except` (skip the bad
row, count it, keep going), or guard `_t_rel`'s `float()` casts. The T08 proof's
"junk values do not raise" case only exercised junk *non-t* columns, so it
missed this.

Covered by `test_unformattable_row_currently_aborts_the_flush` (pins the current
1-row outcome, so the behaviour cannot change unnoticed) and by a matching
`xfail(strict=False)` test asserting the desired 3-row outcome, which flips the
moment the logger is made row-local. `strict=False` so the suite stays green
either way.

## 5. Notes / limitations

* **The mux and adapter share one clock.** Every test injects a fake `clock`
  into the mux; `FakeAdapter` carries no clock of its own, so the vendored
  watchdog cannot trip. A real `RLAdapter` in a test with a fake mux clock must
  be handed the same object (orchestrator note honoured).
* **`make_state()` stamps a fresh `time.time()` per call**, and the mux keys its
  sensor-freshness watchdog on state *values* including `timestamp`. A "stale
  sample" test therefore must pin `timestamp` explicitly; the watchdog test
  documents this inline.
* **The geofence origin is the first valid fix of the session**, so a breach test
  must feed a base-lat tick first, then move — the first tick can never breach.
  Documented inline in the geofence test.
* Slow-disk numbers are **laptop SSD + a simulated 50 ms sleep**, not the Pi SD
  card. Rotation and relative-clock assertions are file-level and hardware
  independent.
* The `no_goal` row is exercised in the `fallback` direction only (empty
  navigator while in `WAYPOINT_NAV`/`STATION_KEEP`); the mux never feeds `None`
  into the policy, asserted via `adapter.calls == []`.
