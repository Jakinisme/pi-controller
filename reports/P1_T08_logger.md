# P1 / T08 — RLLogger

Owner: `src/rl/logger.py` (nothing else touched).

## What was built

`src/rl/logger.py` — `RLLogger` exactly per the frozen interface:

- `COLUMNS` = the 25 frozen columns in the frozen order (header printed in the run
  output below is byte-exact).
- `__init__(log_dir, *, max_bytes=20*1024*1024, flush_interval_s=2.0, clock=time.monotonic)`,
  file `rl_YYYYmmdd_HHMMSS.csv`, rotation to `..._part2.csv`, `..._part3.csv`, ...
- `log(row)` = `deque.append(dict(row))` only. No disk, no lock, never raises.
- `flush()` / `close()` / `__enter__` / `__exit__` / `path` property.
- Booleans as `0`/`1`, floats as `%.6f`, `fault` empty string when `None`,
  `mode`/`source` written verbatim (`waypoint`/`station_keeping`;
  `policy`/`fallback`/`geofence`/`zero`).
- `t_monotonic` = seconds since the **first logged row**. If a row carries its own
  `t_monotonic` (the mux passes `clock()`), that value is re-based against the first
  row; if it does not, the logger's `clock()` is used. Either way row 1 is
  `0.000000`, so `DRL/runtime/infer_replay.py` can use the column as both replay
  clock and estimator timestamp.
- Live file opens on first flush and reopens on rotation; `path` is `None` before
  that and after `close()`. Write errors inside `flush()` are counted in
  `self._errors`, never raised — logging must never take the control loop down.
- Header alone never rotates: a part is only rolled when it already holds at least
  one data row, so a row larger than `max_bytes` still lands (one row per part)
  instead of looping forever.

`ponytail:` notes live in the module (single-file logger, no background thread,
drop-on-unreadable-row, errors counted not raised).

## Replay gap (stated in the module docstring, repeated here)

`DRL/runtime/infer_replay.py` needs `t, lat, lon, heading_deg, speed_mps`
(required) and `yaw_rate_deg_s, course_deg, gps_valid, imu_valid, mag_valid,
est_age_s` (optional), plus `goal_lat, goal_lon, goal_heading_deg, mode,
next_lat, next_lon, next2_lat, next2_lon`.

Our header covers all of those **except** `est_age_s`, `goal_heading_deg`,
`next_lat`, `next_lon`, `next2_lat`, `next2_lon`. So required + most optional
columns replay fine, but **station-keeping heading and the next / next-next
waypoints are NOT replayable from this log** — only the active `goal_lat`/
`goal_lon` is recorded. Adding these columns is a spec change, not a logger
tweak.

The file is written by the **deployment** at runtime. Training never writes it.

## Verification actually run

```
cd E:/buoy-project/pi-control
.venv/Scripts/python.exe C:/Users/AORUS/AppData/Local/hermes/cache/scratch/t08_logger_proof.py
exit code 0
```

Script loads `src/rl/logger.py` with `importlib.util.spec_from_file_location`
because `src/rl/__init__.py` (T04's file, concurrent work) currently fails to
import in this tree state — `src.rl.policy` → `src.rl.obs_np` raises
`ModuleNotFoundError: No module named 'common'`, so `from src.rl.logger import
RLLogger` is blocked until T04's vendored package is complete. Flagged, not
patched (not my file).

Environment: **laptop SSD, not the Pi SD card.** The slow-disk test simulates
50 ms/row with a sleep, it does not measure the real SD card.

Real output:

```
(1) round-trip 25 rows
  [PASS] one csv file  ['rl_20261028_180920.csv']
  [PASS] header exact  t_monotonic,lat,lon,heading_deg,yaw_rate_deg_s,speed_mps,course_deg,gps_valid,imu_valid,mag_valid,goal_lat,goal_lon,mode,wp_idx,n_wp,source,fault,inference_ms,action_raw_surge,action_raw_sway,action_raw_yaw,action_applied_surge,action_applied_sway,action_applied_yaw,thrust_scale
  [PASS] 25 data rows  25 rows
  [PASS] bool as 0/1
  [PASS] None fault -> empty
  [PASS] float %.6f  0.420000
  [PASS] t_monotonic starts at 0
  [PASS] t_monotonic small (relative)  0.000438
  [PASS] filename rl_YYYYmmdd_HHMMSS.csv  rl_20261028_180920.csv
(2) rotation max_bytes=200
  [PASS] _part2 and _part3 exist  ['rl_20261028_180920.csv', 'rl_20261028_180920_part2.csv', 'rl_20261028_180920_part3.csv', 'rl_20261028_180920_part4.csv', 'rl_20261028_180920_part5.csv', 'rl_20261028_180920_part6.csv']
  [PASS] no row lost across rotation  6 rows
  [PASS] rotation degrades to header+1 row per part when a row cannot fit  6 parts, max 470 B, header 280 B
(3) slow write path, 200 rows
  [PASS] total time inside log() < 50 ms  0.748 ms
  [PASS] no row on disk before flush
  [PASS] slow path actually exercised  200 slow writes
  [PASS] all 200 rows present after close  200 rows, close took 10.07s
(4) edge cases + context manager
  [PASS] tiny max_bytes still keeps one row in part1  1 file(s)
  [PASS] flush() while open writes the row
  [PASS] context manager closes
  [PASS] log() after close does not raise
  [PASS] flush() rate-limited then writes after interval
  [PASS] junk float values do not raise  0.000000,,,,,,,0,0,0,,,,,,,,,,,,,,,

==============================================
PASS  one csv file
PASS  header exact
PASS  25 data rows
PASS  bool as 0/1
PASS  None fault -> empty
PASS  float %.6f
PASS  t_monotonic starts at 0
PASS  t_monotonic small (relative)
PASS  filename rl_YYYYmmdd_HHMMSS.csv
PASS  _part2 and _part3 exist
PASS  no row lost across rotation
PASS  rotation degrades to header+1 row per part when a row cannot fit
PASS  total time inside log() < 50 ms
PASS  no row on disk before flush
PASS  slow path actually exercised
PASS  all 200 rows present after close
PASS  tiny max_bytes still keeps one row in part1
PASS  flush() while open writes the row
PASS  context manager closes
PASS  log() after close does not raise
PASS  flush() rate-limited then writes after interval
PASS  junk float values do not raise
SUMMARY: 22/22 PASS -> PASS
NOTE: laptop SSD, not the Pi SD card.
```

## Notes for the other owners

- Mux (T06): pass the raw `clock()` value as `row["t_monotonic"]`; the logger
  re-bases it. Never pre-subtract — double subtraction would give a negative clock.
- `max_bytes=200` is smaller than the 280-byte header, so every part holds
  header + one row (6 rows → 6 files). Rows are never split across files, and
  `_part2`/`_part3` appear as required. With the real 20 MB limit this is
  unreachable.
