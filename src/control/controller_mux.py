"""ControllerMux — the Pi-side safety layer around the trained PPO policy.

`ThrustComputer` (PID) is the default controller and is never modified. The mux
wraps it: it decides per tick whether the vendored ONNX policy runs at all, and
when the policy is rejected it falls back to the *unchanged* PID path. The output
is always an ordinary :class:`ThrustCommand`, so `src/main.py` can swap
`thrust.compute(...)` for `controller.compute(...)` with no other change.

Why this file exists: `RLController` (vendored from DRL, read-only) owns the
policy's own safety layer, but two of its behaviours are wrong for deployment —
its geofence defaults to the *training* 80 m radius, and a geofence breach returns
zeros *before* the slew limiter (CONTRACT note in `p1_brief.md`). The mux therefore
enforces a tighter 20 m geofence itself, latches the system on a breach, and ends
the RL session. See `reports/P1_T06_mux.md` for the full reasoning.

Fault ladder (evaluated every tick, first match wins) is frozen in
`p1_interfaces.md` §ControllerMux; the names below match it verbatim.

Watchdog clock note (important): `FusedState.timestamp` is set from
`time.time()` in `src/sensors/fusion.py`, i.e. it is a **wall-clock** stamp, not a
monotonic sample time. It cannot be compared against the injected monotonic clock
used everywhere else in the control loop. The mux therefore keeps its own
"sensor freshness" timestamp: it records `clock()` whenever the fused state it is
given *changes* (identity changes, or `timestamp`/`lat`/`lon`/`heading` differ).
A frozen state older than `RL_WATCHDOG_S` (by that record) is a watchdog fault.
This does not double-count the policy's own internal watchdog, which compares the
estimator `t` it is handed; the mux still emits `source="fallback"`,
`fault="watchdog"` when its own freshness check fires.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Optional

from src.control.thrust import ThrustCommand
from src.control.navigation import NavigationState
from src.sensors.fusion import FusedState  # re-used, never duplicated

try:  # config is always present in the deployment; keep import soft for tests
    from src import config as _config
except Exception:  # pragma: no cover
    _config = None


log = logging.getLogger("controller_mux")

# Fallback values used only when a name is missing from the config object.
_DEFAULTS = {
    "RL_ENABLED": False,
    "RL_DEFAULT_CONTROLLER": "pid",
    "RL_GEOFENCE_M": 20.0,
    "RL_WATCHDOG_S": 0.5,
    "RL_MAX_INFERENCE_MS": 50.0,
    "RL_MAX_THRUST_SCALE": 1.0,
}

# Ladder fault strings that count toward the 3-strike latch (rule 4).
_LATCH_FAULTS = ("gps", "imu", "mag", "watchdog", "inference", "nan",
                 "slow_inference")
_RL_MODES = (NavigationState.WAYPOINT_NAV, NavigationState.STATION_KEEP)
_LATCH_AFTER = 3
_JITTER_WARN_INTERVAL_S = 5.0


def _cfg(name: str, config: Any) -> Any:
    cfg = _config if config is None else config
    if cfg is None:
        return _DEFAULTS.get(name)
    if isinstance(cfg, dict):
        val = cfg.get(name, _DEFAULTS.get(name))
    else:
        val = getattr(cfg, name, _DEFAULTS.get(name))
    return _DEFAULTS.get(name) if val is None else val


class ControllerMux:
    """Safety layer + decimator selecting between PID and the PPO policy.

    Args:
        thrust: a :class:`ThrustComputer` (PID path, untouched).
        adapter: an :class:`~src.rl.adapter.RLAdapter`-shaped object (needs
            `available`, `step(est, goal)`, `reset()`, `policy_dt_s`,
            `last_inference_ms`; may also expose `meta`/`status`).
        navigator: the :class:`~src.control.navigation.WaypointNavigator` (the
            mux calls ``navigator.stop()`` on a geofence breach).
        gps: the GPS reader, used only to recover the true COG for the policy
            observation (``gps.data.true_course``); optional.
        config: a module/dict exposing the ``RL_*`` names; defaults to
            :mod:`src.config`.
        clock: monotonic time source (injectable for tests/replay).
        logger: plain logger for warnings; defaults to this module's logger.
        rl_logger: an :class:`~src.rl.logger.RLLogger` (or anything with
            ``log(row)``); one row per *policy* step plus one per source change.
    """

    def __init__(self, thrust, adapter, navigator, gps=None, *, config=None,
                 clock=time.monotonic, logger=None, rl_logger=None):
        self.thrust = thrust
        self.adapter = adapter
        self.navigator = navigator
        self.gps = gps
        self._config = config
        self._clock = clock
        self.log = logger or log
        self.rl_logger = rl_logger

        self.selected = "pid"
        self.source = "pid"
        self.fault: Optional[str] = None

        self._request: Optional[str] = None      # last set_controller_request()
        self._latched = False                    # sticky until a new command
        self._fault_streak = 0

        self._policy_dt_s = float(getattr(adapter, "policy_dt_s", 0.1) or 0.1)
        self._last_step_t: Optional[float] = None
        self._last_applied = (0.0, 0.0, 0.0)
        self._last_cmd: Optional[ThrustCommand] = None
        self._last_jitter_warn = 0.0
        self._rl_active = False    # has any policy output been emitted yet?

        self._lat0: Optional[float] = None       # ENU origin of the RL session
        self._lon0: Optional[float] = None
        self._state_key = None                   # identity/values of last state
        self._state_seen_t: Optional[float] = None

        self._last_source_logged = self.source

    # ------------------------------------------------------------- selection
    def set_controller_request(self, name) -> None:
        """Record a controller request: ``"pid"`` or ``"ppo"``.

        Anything else is normalised to ``"pid"`` with one warning. Resets the RL
        session state and clears the latch, so a new command always takes effect.
        """
        if name not in ("pid", "ppo"):
            self.log.warning("Unknown controller request %r -> 'pid'", name)
            name = "pid"
        self._request = name
        self._reset_rl_state()
        self._latched = False
        self._fault_streak = 0

    def reset(self) -> None:
        """Clear RL history + latches. The PID controllers are left untouched."""
        self._reset_rl_state()
        self._latched = False
        self._fault_streak = 0
        self._resolve_selected()
        if self.selected != "ppo":
            self.source = "pid"
            self.fault = None

    def _reset_rl_state(self) -> None:
        self._policy_dt_s = float(getattr(self.adapter, "policy_dt_s", 0.1) or 0.1)
        self._last_step_t = None
        self._last_cmd = None
        self._last_applied = (0.0, 0.0, 0.0)
        self._rl_active = False
        self._lat0 = None
        self._lon0 = None
        self._state_key = None
        self._state_seen_t = None
        try:
            self.adapter.reset()
        except Exception as exc:  # pragma: no cover - reset must never throw
            self.log.warning("adapter.reset() raised: %s", exc)

    def _resolve_selected(self) -> None:
        """Rule 1: PPO only when enabled, requested/defaulted, and available."""
        enabled = bool(_cfg("RL_ENABLED", self._config))
        default_ctl = str(_cfg("RL_DEFAULT_CONTROLLER", self._config) or "pid")
        request = self._request
        wants_ppo = (request == "ppo") or (request is None and default_ctl == "ppo")
        available = bool(getattr(self.adapter, "available", False))
        if enabled and wants_ppo and available:
            self.selected = "ppo"
        else:
            self.selected = "pid"

    # ----------------------------------------------------------------- compute
    def compute(self, nav_output, state: FusedState, now=None) -> ThrustCommand:
        """Decide the command for this tick. Same shape as ``ThrustComputer.compute``."""
        self._policy_dt_s = float(getattr(self.adapter, "policy_dt_s", 0.1) or 0.1)
        if now is None:
            now = self._clock()

        # Rule 2: only autonomous modes may run the policy. IDLE/MANUAL are a
        # straight pass-through, byte-identical and NEVER slew-limited.
        if nav_output.mode not in _RL_MODES:
            cmd = self.thrust.compute(nav_output, state)
            self._last_applied = (cmd.surge, cmd.sway, cmd.yaw)
            self._last_cmd = cmd
            self.source = "pid"
            self.fault = None
            self._update_state_freshness(state, now)
            return cmd

        self._update_state_freshness(state, now)
        self._resolve_selected()

        # Latched (3-strike or geofence): behave as PID, report the latch.
        if self._latched:
            return self._emit_pid(nav_output, state, "fallback", "latched", now)

        # Not selected for RL: PID. (Adapter unavailable while RL enabled+requested
        # lands here too — `_resolve_selected` already forced selected="pid".)
        if self.selected != "ppo":
            enabled = bool(_cfg("RL_ENABLED", self._config))
            available = bool(getattr(self.adapter, "available", False))
            if enabled and self._request == "ppo" and not available:
                # Rule 4 last row: unavailable while the operator asked for PPO.
                self._latched = True
                return self._emit_pid(nav_output, state, "fallback",
                                      "unavailable", now)
            self.source = "pid"
            self.fault = None
            return self._emit_pid(nav_output, state, "pid", None, now)

        # --- fault ladder, first match wins ---------------------------------
        fault = self._sensor_fault(state, now)
        if fault is not None:
            return self._ladder_fallback(nav_output, state, fault, now)

        # Geofence (mux's own tighter radius) BEFORE the policy runs. Rule 5.
        if self._geofence_breached(state):
            self._latched = True
            self.navigator.stop()
            self.source = "zero"
            self.fault = "geofence"
            cmd = self._emit(action=(0.0, 0.0, 0.0), nav_output=nav_output,
                             state=state, now=now, source="zero", fault="geofence")
            return cmd

        # --- decimated policy step (rule 3) --------------------------------
        if self._due_for_step(now):
            result = self._run_policy(nav_output, state, now)
            if result is not None:
                return result
        # in-between tick: repeat the last applied policy command, no inference
        if self._last_cmd is not None:
            return self._emit(action=None, nav_output=nav_output, state=state,
                              now=now, source=self.source, fault=self.fault,
                              repeat=True)

        # No policy result yet (first tick edge): fall back to PID silently.
        self.source = "pid"
        self.fault = None
        return self._emit_pid(nav_output, state, "pid", None, now)

    # ------------------------------------------------------------- ladder bits
    def _sensor_fault(self, state: FusedState, now: float) -> Optional[str]:
        if not getattr(state, "gps_valid", False):
            return "gps"
        if not getattr(state, "imu_valid", False):
            return "imu"
        if not getattr(state, "mag_valid", False):
            return "mag"
        if self._state_seen_t is not None:
            wd = float(_cfg("RL_WATCHDOG_S", self._config))
            if now - self._state_seen_t > wd:
                return "watchdog"
        return None

    def _geofence_breached(self, state: FusedState) -> bool:
        lat = getattr(state, "latitude", None)
        lon = getattr(state, "longitude", None)
        if lat is None or lon is None:
            return False
        if self._lat0 is None:
            # First valid fix of the current RL session establishes the origin.
            self._lat0 = float(lat)
            self._lon0 = float(lon)
            return False
        from src.utils.geo import haversine_distance
        dist = haversine_distance(self._lat0, self._lon0, float(lat), float(lon))
        return dist > float(_cfg("RL_GEOFENCE_M", self._config))

    def _ladder_fallback(self, nav_output, state, fault, now) -> ThrustCommand:
        self._fault_streak += 1
        if self._fault_streak >= _LATCH_AFTER:
            self._latched = True
            self.source = "fallback"
            self.fault = "latched"
            return self._emit_pid(nav_output, state, "fallback", "latched", now)
        return self._emit_pid(nav_output, state, "fallback", fault, now)

    # ------------------------------------------------------------ policy step
    def _due_for_step(self, now: float) -> bool:
        if self._last_step_t is None:
            return True
        return (now - self._last_step_t) >= self._policy_dt_s * 0.9

    def _run_policy(self, nav_output, state, now) -> Optional[ThrustCommand]:
        self._check_jitter(now)
        self._last_step_t = now
        est = self._build_est(state, now)
        goal = self._build_goal(nav_output)
        if goal.get("lat") is None or goal.get("lon") is None:
            # No usable goal (navigator idle/empty while still in an autonomous mode).
            # Feeding None into the policy produces NaN observations, so treat it as a
            # fault for this tick and let PID fly, exactly like a stale sensor.
            return self._ladder_fallback(nav_output, state, "no_goal", now)
        try:
            result = self.adapter.step(est, goal)
        except Exception as exc:
            self.log.warning("adapter.step raised: %s", exc)
            return self._ladder_fallback(nav_output, state, "inference", now)

        action = self._finite(getattr(result, "action", None))
        if action is None:
            return self._ladder_fallback(nav_output, state, "nan", now)

        # Slow inference is checked against the adapter's reported last step.
        max_ms = float(_cfg("RL_MAX_INFERENCE_MS", self._config))
        last_ms = float(getattr(self.adapter, "last_inference_ms", 0.0) or 0.0)
        if last_ms > max_ms:
            return self._ladder_fallback(nav_output, state, "slow_inference", now)

        # Rule 7: thrust scale applies to the POLICY action only, then clip.
        scale = float(_cfg("RL_MAX_THRUST_SCALE", self._config))
        scaled = tuple(max(-1.0, min(1.0, a * scale)) for a in action)

        # Adapter-reported geofence/zero are treated as a latched fault (the
        # vendored geofence bypasses its own slew limit — brief's known bug).
        rl_source = str(getattr(result, "source", "policy"))
        rl_fault = getattr(result, "fault", None)
        if rl_source in ("zero", "geofence"):
            self._latched = True
            self.source = "zero"
            self.fault = rl_fault or rl_source
            return self._emit(action=(0.0, 0.0, 0.0), nav_output=nav_output,
                              state=state, now=now, source="zero",
                              fault=self.fault)

        self._fault_streak = 0
        self._rl_active = True
        self.source = "policy"
        self.fault = rl_fault
        self._emit_log_row(nav_output, state, now, result, scaled)
        return self._emit(action=scaled, nav_output=nav_output, state=state,
                          now=now, source="policy", fault=rl_fault)

    # ---------------------------------------------------------------- logging
    def _emit_log_row(self, nav_output, state, now, result, applied) -> None:
        if self.rl_logger is None:
            return
        try:
            raw = self._finite(getattr(result, "raw_action", None)) or applied
            mode = getattr(nav_output, "mode", NavigationState.IDLE)
            goal = self._build_goal(nav_output)      # the same dict the policy saw
            self.rl_logger.log({
                "t_monotonic": now,
                "lat": getattr(state, "latitude", None),
                "lon": getattr(state, "longitude", None),
                "heading_deg": getattr(state, "heading", None),
                "yaw_rate_deg_s": getattr(state, "yaw_rate", None),
                "speed_mps": getattr(state, "speed", None),
                "course_deg": self._course(state),
                "gps_valid": getattr(state, "gps_valid", False),
                "imu_valid": getattr(state, "imu_valid", False),
                "mag_valid": getattr(state, "mag_valid", False),
                "goal_lat": goal.get("lat"),
                "goal_lon": goal.get("lon"),
                "mode": "station_keeping" if mode == NavigationState.STATION_KEEP
                        else "waypoint",
                "wp_idx": getattr(self.navigator, "current_wp_index", None),
                "n_wp": len(getattr(self.navigator, "waypoints", []) or []),
                "source": "policy",
                "fault": getattr(result, "fault", None),
                "inference_ms": getattr(result, "inference_ms", None),
                "action_raw_surge": raw[0], "action_raw_sway": raw[1],
                "action_raw_yaw": raw[2],
                "action_applied_surge": applied[0], "action_applied_sway": applied[1],
                "action_applied_yaw": applied[2],
                "thrust_scale": float(_cfg("RL_MAX_THRUST_SCALE", self._config)),
            })
            # Mark this source as logged so the _emit() path does not write a
            # second, duplicate row for the same tick.
            self._last_source_logged = "policy"
        except Exception:
            pass  # logging must never raise into the control loop

    def _log_source_change(self, nav_output, state, now, source, fault) -> None:
        """One row on every source change so faults are visible in the CSV.

        Fills the same columns as a policy row (with the real sensor values and
        the applied action) instead of a short row: a sparse row would read as
        gps_valid=0 in the sysid/replay tooling, which is a lie.
        """
        if self.rl_logger is None or source == self._last_source_logged:
            return
        try:
            goal = self._build_goal(nav_output)
            mode = getattr(nav_output, "mode", NavigationState.IDLE)
            # A source change happens *instead* of a policy step, so the action
            # actually sent this tick is only known to the caller; the last one
            # is the best available record and is flagged by `source`.
            last = self._last_applied or (0.0, 0.0, 0.0)
            self.rl_logger.log({
                "t_monotonic": now,
                "lat": getattr(state, "latitude", None),
                "lon": getattr(state, "longitude", None),
                "heading_deg": getattr(state, "heading", None),
                "yaw_rate_deg_s": getattr(state, "yaw_rate", None),
                "speed_mps": getattr(state, "speed", None),
                "course_deg": self._course(state),
                "gps_valid": getattr(state, "gps_valid", False),
                "imu_valid": getattr(state, "imu_valid", False),
                "mag_valid": getattr(state, "mag_valid", False),
                "goal_lat": goal.get("lat"),
                "goal_lon": goal.get("lon"),
                "mode": "station_keeping" if mode == NavigationState.STATION_KEEP
                        else "waypoint",
                "wp_idx": getattr(self.navigator, "current_wp_index", None),
                "n_wp": len(getattr(self.navigator, "waypoints", []) or []),
                "source": source,
                "fault": fault,
                "inference_ms": getattr(self.adapter, "last_inference_ms", None),
                "action_raw_surge": last[0], "action_raw_sway": last[1],
                "action_raw_yaw": last[2],
                "action_applied_surge": last[0], "action_applied_sway": last[1],
                "action_applied_yaw": last[2],
                "thrust_scale": float(_cfg("RL_MAX_THRUST_SCALE", self._config)),
            })
        except Exception:
            pass
        self._last_source_logged = source

    # ------------------------------------------------------------- emit paths
    def _emit_pid(self, nav_output, state, source, fault, now) -> ThrustCommand:
        cmd = self.thrust.compute(nav_output, state)  # untouched PID path
        if self._rl_active:
            # Handover slew only: a policy command was emitted before, so the
            # PID output must not step. A pure-PID deployment is byte-exact.
            applied = self._slew(cmd.surge, cmd.sway, cmd.yaw, now)
        else:
            applied = (cmd.surge, cmd.sway, cmd.yaw)
        self._last_applied = applied
        self._last_cmd = ThrustCommand(surge=applied[0], sway=applied[1],
                                       yaw=applied[2], timestamp=cmd.timestamp)
        self.source = source
        self.fault = fault
        self._log_source_change(nav_output, state, now, source, fault)
        return self._last_cmd

    def _emit(self, action, nav_output, state, now, source, fault,
              repeat=False) -> ThrustCommand:
        if action is None and repeat:
            action = self._last_applied
        applied = self._slew(action[0], action[1], action[2], now)
        self._last_applied = applied
        self._last_cmd = ThrustCommand(surge=applied[0], sway=applied[1],
                                       yaw=applied[2], timestamp=time.time())
        self.source = source
        self.fault = fault
        self._log_source_change(nav_output, state, now, source, fault)
        return self._last_cmd

    def _slew(self, surge, sway, yaw, now) -> tuple:
        """Rule 6: clamp each axis within rate_limit_per_s * policy_dt of the last.

        ``dt`` is the policy step (2.0/s x 0.1 s = 0.2), not the wall gap between
        emissions, so a long fault does not license a big handover jump.
        """
        max_step = self._rate_limit_per_s() * self._policy_dt_s
        out = []
        for prev, want in zip(self._last_applied, (surge, sway, yaw)):
            delta = max(-max_step, min(max_step, want - prev))
            out.append(prev + delta)
        return tuple(out)

    def _rate_limit_per_s(self) -> float:
        safety = getattr(self.adapter, "safety", None)
        rate = None
        if isinstance(safety, dict):
            rate = safety.get("rate_limit_per_s")
        if rate is None:
            meta = getattr(self.adapter, "meta", None)
            if isinstance(meta, dict):
                rate = (meta.get("safety") or {}).get("rate_limit_per_s")
        return float(rate) if rate else 2.0

    # ---------------------------------------------------------------- helpers
    def _finite(self, action) -> Optional[tuple]:
        if action is None:
            return None
        try:
            vals = tuple(float(a) for a in action)
        except (TypeError, ValueError):
            return None
        if len(vals) < 3:
            return None
        vals = vals[:3]
        if not all(math.isfinite(v) for v in vals):
            return None
        return vals

    def _build_est(self, state: FusedState, now: float) -> dict:
        gps_valid = bool(getattr(state, "gps_valid", False))
        return {
            "lat": getattr(state, "latitude", 0.0),
            "lon": getattr(state, "longitude", 0.0),
            "heading_deg": getattr(state, "heading", 0.0),
            "yaw_rate_deg_s": getattr(state, "yaw_rate", 0.0),
            "speed_mps": getattr(state, "speed", 0.0),
            "course_deg": self._course(state),
            "gps_valid": gps_valid,
            "imu_valid": bool(getattr(state, "imu_valid", False)),
            "mag_valid": bool(getattr(state, "mag_valid", False)),
            "sensor_ok": gps_valid and bool(getattr(state, "imu_valid", False))
                         and bool(getattr(state, "mag_valid", False)),
            "t": now,
        }

    def _course(self, state: FusedState) -> Optional[float]:
        try:
            return float(self.gps.data.true_course)
        except Exception:
            return getattr(state, "heading", None)

    def _build_goal(self, nav_output) -> dict:
        wp = self._current_waypoint()
        if wp is None:
            return {"lat": None, "lon": None, "mode": "waypoint"}
        goal = {"lat": float(wp.latitude), "lon": float(wp.longitude),
                "mode": "waypoint"}
        hdg = getattr(wp, "heading_hold", None)
        if hdg is not None:
            goal["heading_deg"] = hdg
        return goal

    def _current_waypoint(self):
        wp = getattr(self.navigator, "current_waypoint", None)
        if callable(wp):
            try:
                wp = wp()
            except Exception:
                wp = None
        if wp is None:
            wps = getattr(self.navigator, "waypoints", None) or []
            idx = getattr(self.navigator, "current_wp_index", 0)
            if 0 <= idx < len(wps):
                wp = wps[idx]
        return wp

    def _check_jitter(self, now: float) -> None:
        if self._last_step_t is None:
            return
        gap = now - self._last_step_t
        if self._policy_dt_s <= 0:
            return
        dev = abs(gap - self._policy_dt_s) / self._policy_dt_s
        if dev > 0.3 and now - self._last_jitter_warn >= _JITTER_WARN_INTERVAL_S:
            self._last_jitter_warn = now
            self.log.warning("Policy call jitter: gap %.1f ms (target %.1f ms, "
                             "%.0f%% off)", gap * 1000.0, self._policy_dt_s * 1000.0,
                             dev * 100.0)

    def _update_state_freshness(self, state: FusedState, now: float) -> None:
        """Record when the fused state last *changed* (see the watchdog note).

        Keyed on values, never on ``id(state)``: CPython reuses object addresses
        after GC, so an id-based key would false-negative when the caller builds
        a fresh FusedState per tick. An unchanged sample == a stale sample.
        """
        key = (getattr(state, "timestamp", None),
               getattr(state, "latitude", None), getattr(state, "longitude", None),
               getattr(state, "heading", None), getattr(state, "speed", None),
               getattr(state, "yaw_rate", None),
               getattr(state, "gps_valid", None), getattr(state, "imu_valid", None),
               getattr(state, "mag_valid", None))
        if key != self._state_key:
            self._state_key = key
            self._state_seen_t = now
