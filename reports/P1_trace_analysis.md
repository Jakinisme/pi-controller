# P1 — Trace analysis of the failed RL smoke run

Source CSV: `logs/rl_smoke/rl_20261028_233320.csv` (newest file, 145 rows, t = 0.000 → 14.350 s).
Every number below is computed from that CSV. Mock-run data only — nothing here is a claim about the real vessel.

## Method

- ENU (east, north) per row, relative to the **first row's fix** `lat0=-6.896900, lon0=112.052100`, equirectangular, `R=6371000`, east scaled by `cos(lat0)=0.992764`.
- Goal is constant for the whole run: `-6.896450, 112.052100` → ENU `(east 0.000, north 50.038)` m. Bearing to goal from the start is **000.0° (due north)**; straight-line distance **50.038 m**.
- `wp_idx = 0` and `n_wp = 4` on every row. Only leg 0 was ever executed — there is no second leg in this log.
- `sources`: `policy` 144, `zero` 1 (last row, `fault=geofence`). `thrust_scale = 1.000000` throughout, and `action_raw_* == action_applied_*` on all 145 rows (no clamping or scaling in effect).

## 1. Numeric trace (1 s samples)

| t [s] | east [m] | north [m] | range_to_goal [m] | heading [°] | surge | sway | yaw |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | +0.000 | +0.000 | 50.038 | 0.000 | +0.2000 | -0.2000 | +0.2000 |
| 1.00 | +0.110 | +0.334 | 49.704 | 15.000 | +1.0000 | +0.3748 | +0.8066 |
| 2.00 | +0.552 | +1.334 | 48.707 | 28.497 | +1.0000 | +0.9086 | +0.5090 |
| 3.00 | +1.104 | +2.557 | 47.493 | 36.724 | +1.0000 | +0.9264 | +0.3009 |
| 4.00 | +1.766 | +3.892 | 46.180 | 41.109 | +1.0000 | +0.9434 | +0.1318 |
| 5.00 | +2.539 | +5.115 | 44.994 | 42.570 | +1.0000 | +0.9547 | +0.0115 |
| 6.00 | +3.422 | +6.449 | 43.723 | 42.236 | +1.0000 | +0.9589 | -0.0394 |
| 7.00 | +4.195 | +7.784 | 42.462 | 41.351 | +1.0000 | +0.9597 | -0.0453 |
| 8.00 | +4.968 | +9.007 | 41.331 | 40.492 | +1.0000 | +0.9601 | -0.0402 |
| 9.00 | +5.740 | +10.341 | 40.109 | 39.711 | +1.0000 | +0.9609 | -0.0386 |
| 10.00 | +6.403 | +11.675 | 38.893 | 38.926 | +1.0000 | +0.9621 | -0.0408 |
| 11.00 | +7.175 | +13.121 | 37.608 | 38.069 | +1.0000 | +0.9637 | -0.0452 |
| 12.00 | +7.838 | +14.455 | 36.435 | 37.119 | +1.0000 | +0.9651 | -0.0508 |
| 13.00 | +8.500 | +15.790 | 35.287 | 36.036 | +1.0000 | +0.9664 | -0.0588 |
| 14.00 | +9.052 | +17.235 | 34.029 | 34.768 | +1.0000 | +0.9676 | -0.0693 |
| 14.35 (latch) | +9.273 | +17.680 | 33.660 | 34.274 | +0.8000 | +0.7678 | 0.0000 |

Supporting per-row state (10-row decimation):

| t [s] | bearing_to_goal [°] | heading [°] | heading_err [°] | cross_track [m] | speed [m/s] | yaw_rate [°/s] |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.000 | +0.00 | +0.000 | 0.000 | +0.000 |
| 1.00 | 359.873 | 15.000 | -15.13 | +0.110 | 0.809 | +16.644 |
| 2.00 | 359.351 | 28.497 | -29.15 | +0.552 | 1.245 | +10.703 |
| 3.00 | 358.668 | 36.724 | -38.06 | +1.104 | 1.419 | +6.412 |
| 4.00 | 357.808 | 41.109 | -43.30 | +1.766 | 1.485 | +2.922 |
| 5.00 | 356.765 | 42.570 | -45.81 | +2.539 | 1.511 | +0.411 |
| 6.00 | 355.511 | 42.236 | -46.73 | +3.422 | 1.521 | -0.740 |
| 7.00 | 354.330 | 41.351 | -47.02 | +4.195 | 1.525 | -0.914 |
| 8.00 | 353.097 | 40.492 | -47.40 | +4.968 | 1.527 | -0.812 |
| 9.00 | 351.772 | 39.711 | -47.94 | +5.740 | 1.527 | -0.770 |
| 10.00 | 350.525 | 38.926 | -48.40 | +6.403 | 1.527 | -0.808 |
| 11.00 | 349.001 | 38.069 | -49.07 | +7.175 | 1.527 | -0.896 |
| 12.00 | 347.578 | 37.119 | -49.54 | +7.838 | 1.528 | -1.003 |
| 13.00 | 346.061 | 36.036 | -49.98 | +8.500 | 1.528 | -1.157 |
| 14.00 | 344.573 | 34.768 | -50.20 | +9.052 | 1.528 | -1.363 |

Heading sign convention: heading increases from 0° to 42.57°, cross-track is positive (+east of the north-running goal line), so heading turns **toward east** and positive yaw = turn to starboard/east in this log.

## 2. Is range_to_goal decreasing?

- 1 s samples: 15; consecutive pairs: 14; pairs where range decreases: **14/14 = 1.000**. Monotone decreasing at 1 s resolution — no reversal, no loitering.
- Range itself: **50.038 → 33.660 m**, i.e. 16.378 m closed in 14.35 s. Path length travelled **21.317 m**, net displacement **19.964 m**, so the path is not wildly inefficient (path/displacement = 1.068).
- Cross-track from the straight line start→goal: `max |XTE| = 9.273 m` at t = 14.20 s (end of run); XTE grows essentially monotonically +0.110 → +9.273 m. Along-track progress at end: 17.680 m of the 50.038 m leg (**35.3 %**).

So the vessel did **not** fail by standing still or by ranging backwards. It made real, steady progress along the leg while drifting steadily to the east of the geodesic. Closed-loop in the weak sense (range improves), but never converged on the bearing: heading error goes -15° → **-50.2°** and stays there.

## 3. Command integrals, saturation, circling

Trapezoid integration over 0 → 14.350 s (raw and applied are identical):

| component | ∫ dt | mean | mean abs |
|---|---:|---:|---:|
| applied surge | 14.1853 | **+0.9885** | 0.9885 |
| applied sway | 12.2459 | **+0.8534** | 0.9095 |
| applied yaw | 1.7021 | +0.1186 | 0.1769 |

- **Surge is saturated**: 140/145 rows (96.6 %) have `|surge| >= 0.999`, and mean surge = 0.9885. Sway is at 0.9678 by the end — it is heading for saturation too but never hit 0.999 (0/145 rows).
- Yaw is **not** saturated (0/145 rows ≥ 0.999) and is not large in the mean: +0.1186.
- Mean yaw is **positive** (+0.1186) and net heading change is **+34.274°** over the run (0.0952 turns), unwrapped monotonically increasing — but the rotation is not uniform: yaw command starts at +0.98 and decays to ~0 by t = 5 s, then goes slightly negative (-0.04 … -0.07) and stays there, while the observed yaw_rate is -0.7 … -1.4 °/s sustained.
- Circle-from-speed-and-rate: mean speed 1.3901 m/s, mean yaw_rate +2.3587 °/s (ω = 0.04117 rad/s) → implied radius **33.768 m**. That mean yaw rate is contaminated by the launch transient (see below); the late-run numbers are a much cleaner circle: speed ≈ 1.528 m/s with yaw_rate ≈ -0.9 °/s → radius ≈ 97 m.
- **Verdict on "is it circling": no, not a tight circle.** Net heading change is only 34.3° over 14.35 s, i.e. ~2.4 °/s average, with a large transient at t < 3 s and a steady-state residual of roughly -0.9 °/s. In the last 7 s it is a wide, slow **starboard→port drift** (heading 42.57° → 34.27°), not a loop.

Anomaly worth naming: the policy's yaw command is ~0 from t ≥ 5 s (mean -0.04, i.e. 4 % of authority), yet the vessel keeps turning at -0.8 … -1.4 °/s and drifting east. Whatever is generating that turn, it is **not** coming from the commanded yaw channel — the plant dynamics / sway coupling / mock model is producing it. Note `corr(sway, yaw) = -0.706`: sway and yaw are moving in opposition for most of the run (first 5 s sway was negative while yaw was positive; after t ≈ 3 s sway is ~+0.96 while yaw is ~0).

## 4. First 0.5 s vs last 0.5 s before the latch

| window | n rows | applied surge | applied sway | applied yaw | mean heading | mean range |
|---|---:|---|---|---|---:|---:|
| first 0.00–0.40 s | 5 | mean +0.6005 (0.20 … 1.00) | mean **-0.5655** (-0.8254 … -0.2000) | mean **+0.5998** (0.20 … 0.9969) | 1.600° | 50.038 m |
| last 13.90–14.35 s | 6 | mean +0.9667 (0.80 … 1.00) | mean **+0.9344** (+0.7678 … +0.9678) | mean **-0.0586** (-0.0722 … 0.0000) | 34.569° | 33.885 m |

The commands are **not** identical across the run — but they are also not reacting to the navigation error.

- Sway flips sign (-0.57 → +0.93 mean), yaw decays (+0.60 → -0.06). So this is not a frozen constant command.
- But drive this against the state: heading went 0.0° → 34.27°, range closed 50.04 → 33.66 m, bearing to goal swung 0.0° → 344.6°, heading error grew -15° → -50.2°, and **surge stayed at 1.0** the whole time (140/145 rows pinned at ≥ 0.999).
- Late-run values are also nearly static internally: ten consecutive late rows have sway +0.9678 → +0.9676 and yaw -0.0713 → -0.0722, i.e. the policy output changes by < 1e-3 per 100 ms while the heading changes by ~0.14°/100 ms. That is a command trajectory that is effectively a function of elapsed run time, not of the heading error.
- The strong signature is the **inference_ms** column: uniform 0.35–0.60 ms with mean 0.398 ms and total 5.716 ms over 144 policy rows — consistent with a fixed/cheap forward pass, and consistent with the observation that nothing about the command responds to a 50° heading error.

**Conclusion stated with numbers:** the trajectory is *not* open-loop in the strict sense (commands do change over the run), but at the level that matters it is **non-reactive**: a 50° sustained heading error produces surge = 1.000 (saturated) with yaw ≈ -0.04 (4 % of authority). The policy never commands a turn toward the goal line. The vessel closes range to 33.66 m along a curving path while slowly rotating from 0° to 34°, then the mux's own 20 m geofence (distance from the first fix of the session = **19.964 m** at the latch row) latches it to `source=zero, fault=geofence` at t = 14.350 s, and the logged row shows the zeroed command (yaw forced to 0.000000).

## 5. What physically happened, in one paragraph

Buoy starts at the origin, goal 50.038 m due north, 4 waypoints, all sensors valid. Policy outputs a launch transient (surge ramping 0.20 → 1.00 in 1 s, yaw +0.60 mean, sway -0.57 mean) that turns the hull to starboard from 0° to 42.6° by t = 5 s with a peak yaw rate of +19.94 °/s at t = 0.50 s. From t ≈ 5 s onward the policy holds surge at the 1.0 rail, commands sway ≈ +0.96 and yaw ≈ -0.04, and the hull coasts at 1.528 m/s while slowly turning back to port (yaw rate -0.7 … -1.4 °/s). Range to goal decreases every single second (50.04 → 33.66 m) but the vessel tracks 9.27 m east of the goal line with bearing error growing from -15° to -50°. At 19.96 m from the first fix the mux's 20 m geofence fires, latches, and zeroes the command at t = 14.350 s. Waypoints cleared: 0/4.

## Caveat

This is the mock run from `logs/rl_smoke/`. The eastward drift with near-zero commanded yaw suggests the unmodelled turn originates in the simulator/plant model, not in the yaw command channel — that is a property of the mock, and must be reproduced on hardware before any conclusion is drawn about the real vessel.
