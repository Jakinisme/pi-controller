# P1_smoke — mock-hardware smoke run of the RL control path

Owner: T11. Files: `tools/mock_smoke.py`, `docs/RL_INTEGRATION.md`, this report.
Nothing else touched.

**MOCK ONLY.** No vessel, no hardware, no GPS, no serial, no Firebase credentials,
no network. `hardware_mock.install()` fakes `RPi.GPIO`/`pigpio`/`smbus2`/`serial`
and `FirebaseBridge(enabled=False)` makes sure nothing is transmitted. The vessel
is a stand-in first-order speed model written inside `tools/mock_smoke.py`. Every
number below describes *this laptop* and *that stand-in model*; none of it says
anything about the real ASV. It shows the wiring runs and the safety layer never
emits an illegal command. That is all it shows.

---

## 1. What the smoke does

`tools/mock_smoke.py` (`--seconds N`, default 60):

1. `install()`s the hardware mock, then imports `src.*`.
2. Builds the **real** `ASVSystem()` with `FirebaseBridge(enabled=False)`.
3. Forces `src.config.RL_ENABLED = True`, `RL_DEFAULT_CONTROLLER = "ppo"` **for
   this process only** — the shipped default in `src/config.py` stays
   `RL_ENABLED=False` / `"pid"`.
4. Walks a scripted square mission — 4 waypoints, ~50 m legs around the mock home
   `-6.8969 / 112.0521` — through the real `ControllerMux.compute()` at 20 Hz,
   `set_controller_request("ppo")`.
5. Advances lat/lon/heading from the **applied** `ThrustCommand` with a simple
   first-order speed model (see §4), keeping `gps_valid`/`imu_valid`/`mag_valid`
   True every tick, and stamps `gps.data.true_course` from the model heading.
6. Prints the source and fault histograms, `inference_ms` p50/p95/max, control-loop
   period p50/p95/max, waypoints reached and the CSV path.
7. Runs **the same mission again under PID** as a control, so the report shows
   whether the rig/model can complete the square at all.
8. Exits non-zero if any emitted command component is non-finite or outside
   `[-1, 1]`.

## 2. Real output (run for real, 60 s)

```
$ cd E:/buoy-project/pi-control
$ .venv/Scripts/python.exe tools/mock_smoke.py --seconds 60
====================================================================
RL MOCK-HARDWARE SMOKE  (MOCK: no vessel, no hardware, no GPS, no net)
====================================================================
tick rate : 20 Hz  (policy dt 0.1 s)  adapter_available=True
-- ppo controller --
  duration / ticks : 14.40 s / 289
  sources          : {'policy': 288, 'zero': 1}
  faults           : {'': 288, 'geofence': 1}
  waypoints reached: 0/4   final nav state: IDLE
  inference_ms     : p50=0.308 p95=0.389 max=0.513 (n=288)
  loop period p50/p95/max: 49.99/50.44/50.80 ms (target 50 ms)
-- pid controller (same mission, control run) --
  duration / ticks : 59.95 s / 1200
  sources          : {'pid': 1200}
  faults           : {'': 1200}
  waypoints reached: 1/4   final nav state: WAYPOINT_NAV
  loop period p50/p95/max: 50.01/50.36/50.58 ms (target 50 ms)
csv : E:\buoy-project\pi-control\logs\rl_smoke\rl_20261028_200019.csv
--------------------------------------------------------------------
MOCK: no vessel, no hardware, no network. Numbers describe this laptop
and a stand-in physics model, NOT the real ASV.
--------------------------------------------------------------------
PASS: every emitted command finite and within [-1,1]
exit=0
```

The CSV is the real `RLLogger` output (25 frozen columns, 145 rows: policy rows
plus source-change rows). Header and first row:

```
t_monotonic,lat,lon,heading_deg,yaw_rate_deg_s,speed_mps,course_deg,gps_valid,imu_valid,mag_valid,goal_lat,goal_lon,mode,wp_idx,n_wp,source,fault,inference_ms,action_raw_surge,...
0.000000,-6.896900,112.052100,0.000000,0.000000,0.000000,0.000000,1,1,1,-6.896450,112.052100,waypoint,0,4,policy,,0.537200,0.200000,...
0.100236,-6.896900,112.052100,0.400000,4.000000,0.028444,0.400000,1,1,1,-6.896450,112.052100,waypoint,0,4,policy,,0.299800,0.400502,...
```

`inference_ms` here is the adapter wall time on **this laptop** (x86). It is *not*
a Pi 3B number; measure that on the Pi (`tools/bench_inference.py`, see
`reports/P1_T05_adapter.md` §4). `RL_MAX_INFERENCE_MS = 50` is an unvalidated
deployment threshold, and this mock does not change that.

## 3. Findings (read these before trusting a green run)

1. **The policy completed 288 policy steps with no fault other than the geofence,
   and every emitted command was finite and inside `[-1, 1]`.** The plumbing and
   the safety layer work end to end.
2. **The PPO path did NOT reach a waypoint on this mock and latched on the 20 m
   geofence after 14.4 s.** `RLController`/the mux emitted a sustained saturated
   command (surge→+1, sway→+0.85, yaw→+0.55) and the stand-in model drove tight
   circles instead of converging, so the vessel wandered past `RL_GEOFENCE_M`
   and the mux did the right thing: zero thrust, `navigator.stop()`, latch
   (`source='zero'`, `fault='geofence'`). **This is the safety layer working, not
   a bug in it.** Whether R7 tracks in the DRL simulator is a separate question
   answered by the simulator, not by this mock.
3. **PID flew the same mission on the same model and reached waypoint 1/4 in
   60 s** (2.0 m arrival radius, ~35 s per 50 m leg at 1.5 m/s). This is the
   control that shows the rig and the model *can* complete a leg — so finding 2
   is about the policy/model interaction, not about the harness being unable to
   make waypoints advance.
4. **The mux never populates `next_lat`/`next_lon`/`next2_lat`/`next2_lon` in the
   goal** (`src/control/controller_mux.py::_build_goal` sets only
   `lat`/`lon`/`mode`/`heading_deg`). The later waypoints are absent for *every*
   waypoint, not only at the last one as the brief's known-limits list implies.
   `reported, not fixed — not T11's file`. An experiment that injected next/next2
   into the goal still circled, so this is not the cause of finding 2; it remains
   an observation-vs-training gap worth a look by T06.
5. **`src/config.py` was not modified.** The smoke forces `RL_ENABLED=True` in
   memory only. The shipped default and the `pid` default are unchanged.

## 4. The stand-in model (so the numbers above can be read honestly)

`SimpleVessel`, `MAX_SPEED_MPS=1.5`, first-order lag `TAU_S=1.0 s`,
`YAW_RATE_DEG_S=20` at full yaw command, lateral gain `SWAY_GAIN=0.2` (a hull
slides sideways far less than it moves forward). Body→ENU uses `psi` = heading
degrees clockwise from north, `+surge` forward / `+sway` starboard — the same
convention as `src/control/thrust.py` and the firmware `MIX_MATRIX`. The model
exists so the waypoints have a chance to advance; it is deliberately crude and is
**not** the DRL simulator and **not** the vessel.

## 5. Commands run

| command | result |
|---|---|
| `.venv/Scripts/python.exe tools/mock_smoke.py --seconds 60` | exit 0, output in §2 |
| `.venv/Scripts/python.exe tools/mock_smoke.py --seconds 3` | exit 0 (short sanity run) |
| `.venv/Scripts/python.exe -m pytest -q` | 84 passed, 9 failed — all 9 in `tests/test_rl_faults.py` / `tests/test_rl_logger.py` (T10's files, pre-existing, not touched here) |
