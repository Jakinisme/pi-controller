"""Black-box CSV logging of RL decision rows for sysid and replay.

(i) **What it is for.** The CSV this module writes is the ground-truth trace of
    the vessel's control loop. It feeds the bench system-identification work
    (bollard-pull thrust curves, yaw/decay tests) and offline replay of a real
    run, so a controller change can be diffed against the same input sequence.

(ii) **Replay gap.** ``DRL/runtime/infer_replay.py`` consumes this file. The
    frozen header below covers its required columns (``t, lat, lon,
    heading_deg, speed_mps``) and most optional ones (``yaw_rate_deg_s,
    course_deg, gps_valid, imu_valid, mag_valid``), but it is **missing**
    ``est_age_s``, ``goal_heading_deg``, ``next_lat``, ``next_lon``,
    ``next2_lat`` and ``next2_lon``. Consequence, stated plainly: replay
    reconstructs the sensor stream fine, but station-keeping heading goals and
    the next / next-next waypoints are **not replayable** from this log. Only
    the currently active goal (``goal_lat``/``goal_lon``) is known.

(iii) **Who writes it.** This file is written by the *deployment* at runtime
    (``RLLogger`` constructed on the Pi / in the control loop). It is never
    written by training; training lives in the DRL repo and has its own trace
    format. Nothing here reads the DRL repo.

Design: ``log()`` is a pure enqueue onto a ``collections.deque``. Every byte
hits disk inside ``flush()`` / ``close()``, which the control loop calls from a
background task or at shutdown. A slow SD card therefore cannot stall a control
tick.

ponytail: single-owner, single-file logger — no rotation thread, no async I/O,
no log-level plumbing. Add a background flusher task only if a deployment turns
out to call ``flush()`` from the hot path.
"""

import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

__all__ = ["RLLogger"]

_T_FORMAT = "%.6f"
_FLAGS = ("gps_valid", "imu_valid", "mag_valid")
_ACTION_COLS = (
    "action_raw_surge", "action_raw_sway", "action_raw_yaw",
    "action_applied_surge", "action_applied_sway", "action_applied_yaw",
)


class RLLogger:
    """Non-blocking CSV writer for one RL decision stream."""

    COLUMNS = (
        "t_monotonic", "lat", "lon", "heading_deg", "yaw_rate_deg_s",
        "speed_mps", "course_deg", "gps_valid", "imu_valid", "mag_valid",
        "goal_lat", "goal_lon", "mode", "wp_idx", "n_wp", "source", "fault",
        "inference_ms",
        "action_raw_surge", "action_raw_sway", "action_raw_yaw",
        "action_applied_surge", "action_applied_sway", "action_applied_yaw",
        "thrust_scale",
    )

    def __init__(self, log_dir, *, max_bytes=20 * 1024 * 1024,
                 flush_interval_s=2.0, clock=time.monotonic):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = int(max_bytes)
        self._flush_interval_s = float(flush_interval_s)
        self._clock = clock

        self._queue = deque()
        self._t0 = None            # clock() of the first logged row
        self._last_flush = clock()
        self._part = 1
        self._fh = None
        self._bytes = 0
        self._header_bytes = 0
        self._stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._closed = False
        self._errors = 0           # silently swallowed write errors

    # ------------------------------------------------------------------ paths
    @property
    def path(self):
        """Current CSV file, or None before the first flush / after close."""
        if self._fh is None:
            return None
        return Path(self._fh.name)

    def _name(self, part):
        suffix = "" if part == 1 else f"_part{part}"
        return self._dir / f"rl_{self._stamp}{suffix}.csv"

    # ----------------------------------------------------------------- inputs
    def log(self, row):
        """Enqueue one row. Never blocks, never touches disk, never raises."""
        try:
            self._queue.append(dict(row))
        except Exception:
            pass  # ponytail: a row that cannot even be copied is dropped

    # ------------------------------------------------------------------ flush
    def flush(self):
        """Write the queue to disk now. Honours ``flush_interval_s`` for repeat calls.

        ``close()`` calls ``_write_all()`` directly, so shutdown always persists.
        """
        if self._closed or not self._queue:
            return
        if self._clock() - self._last_flush < self._flush_interval_s:
            return
        try:
            self._write_all()
        except Exception:
            # ponytail: logging must never take the control loop down; count instead
            self._errors += 1

    def _write_all(self):
        now = self._clock()
        while self._queue:
            row = self._queue.popleft()
            line = self._format(row)
            need = len(line.encode("utf-8"))
            if self._fh is None:
                self._open_next()
            elif self._bytes + need > self._max_bytes and self._bytes > self._header_bytes:
                self._close_file()
                self._open_next()
            self._fh.write(line)
            self._bytes += need
        self._fh.flush()
        self._last_flush = now

    def _open_next(self):
        p = self._name(self._part)
        self._fh = open(p, "w", encoding="utf-8", newline="")
        self._bytes = 0
        header = ",".join(self.COLUMNS) + "\n"
        self._fh.write(header)
        self._header_bytes = len(header.encode("utf-8"))
        self._bytes = self._header_bytes
        self._part += 1

    def _close_file(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                self._errors += 1
            self._fh = None

    # --------------------------------------------------------------- encoding
    def _format(self, row):
        out = []
        for col in self.COLUMNS:
            val = row.get(col)
            if col == "t_monotonic":
                t = self._t_rel(row.get("t_monotonic"))
                out.append(_T_FORMAT % t)
            elif col in _FLAGS:
                out.append("1" if val else "0")
            elif col in ("fault", "mode", "source"):
                out.append("" if val is None else str(val))
            elif col in ("wp_idx", "n_wp"):
                out.append("" if val is None else str(int(val)))
            else:  # every remaining column is a float
                try:
                    out.append("" if val is None else _T_FORMAT % float(val))
                except (TypeError, ValueError):
                    out.append("")
        return ",".join(out) + "\n"

    def _t_rel(self, raw):
        """Seconds since the first logged row.

        A row may carry its own ``t_monotonic`` (the mux passes ``clock()``);
        when it does not, the logger's own clock is used. Either way the first
        row becomes t=0.
        """
        t = self._clock() if raw is None else raw
        if self._t0 is None:
            self._t0 = t
            return 0.0
        return float(t) - float(self._t0)

    # ------------------------------------------------------------------ close
    def close(self):
        """Flush everything still queued, then close the file. Idempotent."""
        if self._closed:
            return
        try:
            self._write_all()
        except Exception:
            self._errors += 1
        self._closed = True
        self._close_file()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
