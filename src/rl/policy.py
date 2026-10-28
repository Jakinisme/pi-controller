"""ONNX policy controller for the Pi 3B — CONTRACT.md §4.6, §4.8.

numpy + onnxruntime only (no torch). `RLController.step(estimator_state, goal)`
returns the body command `(surge, sway, yaw)` in [-1, 1] that the ESP32 already
mixes today, wrapped in the safety layer:

    validity gate (CONTRACT §4.4 `gps_valid` / `sensor_ok`, NOT the legacy
    `has_fix` of pi-control/src/sensors/gps.py) -> watchdog -> geofence
    -> policy -> clip -> slew-rate limit -> PID fallback hook

Fault policy:
    no GPS fix / sensor stale / watchdog timeout -> PID fallback (callable injected
        by the caller; `None` -> zero command)
    geofence breach -> zero command (the PID would keep driving outward)
    non-finite policy output -> zero command
The slew-rate limit is applied to fallback output too, so a handover is not a step.

TODO(H7/CONTRACT T4): `u, v` are estimated from SOG/COG/heading by
`common/estimator.py`; there is no body-velocity sensor on the real vessel.
TODO: waypoints after the current goal (`next_lat/next_lon`, `next2_lat/next2_lon`)
must be supplied by the navigator; while absent those obs components are 0.0 and the
policy sees a weaker signal than in training (last-waypoint case).
"""

import json
import time

import numpy as np

from src.rl.estimator import estimate_body_velocity
from src.rl.frames import heading_deg_to_psi, latlon_to_enu
from src.rl.obs_np import DEFAULT_CLIP_NORM, ObsBuilder

__all__ = ["RLController", "DEFAULT_SAFETY"]

DEFAULT_SAFETY = {
    "clip": 1.0,
    "rate_limit_per_s": 2.0,
    "geofence_m": 80.0,
    # GPS loss is NOT a local timeout: the caller passes the CONTRACT §4.4
    # `gps_valid` / `sensor_ok` flags (the glue owns staleness, sensors.py decides it).
    # A stale-fix timeout here would double-count and disagree with the sim.
    "watchdog_s": 0.5,
}

_MODES = {"waypoint": 0.0, "station_keeping": 1.0}


class RLController:
    """Deterministic ONNX policy + safety wrapper, single sample per call."""

    def __init__(self, onnx_path=None, meta_path=None, fallback=None, *,
                 onnx_bytes=None, session=None, obs_cfg=None, safety=None,
                 min_speed_mps=0.3, decay_tau_s=1.0, clock=time.monotonic):
        """Args:
            onnx_path, meta_path: exported model + `policy_meta.json` (CONTRACT §4.8).
            fallback: `callable(estimator_state, goal) -> (surge, sway, yaw) | None`,
                normally `baseline_pid`-derived. `None` -> zeros on fault.
            onnx_bytes / session: alternative model injection (tests, in-memory).
            obs_cfg: dict overriding `obs_offset` / `obs_scale` / `obs_clip_norm`.
            safety: dict overriding `DEFAULT_SAFETY` keys.
            min_speed_mps, decay_tau_s: passed to `common/estimator.py`.
            clock: monotonic time source (injectable for replay/tests).
        """
        meta = {}
        if meta_path is not None:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        self.meta = meta

        if session is None:
            import onnxruntime as ort

            if onnx_bytes is None and onnx_path is None:
                raise ValueError("need onnx_path, onnx_bytes or session")
            session = ort.InferenceSession(
                onnx_bytes if onnx_bytes is not None else onnx_path,
                providers=["CPUExecutionProvider"],
            )
        self._session = session
        self._input_name = meta.get("input_name") or session.get_inputs()[0].name
        self._output_name = meta.get("output_name") or session.get_outputs()[0].name

        obs_cfg = obs_cfg or {}
        self._obs = ObsBuilder(
            offset=obs_cfg.get("offset", meta.get("obs_offset")),
            scale=obs_cfg.get("scale", meta.get("obs_scale")),
            clip_norm=obs_cfg.get("clip_norm", meta.get("obs_clip_norm", DEFAULT_CLIP_NORM)),
            k_history=int(meta.get("k_history", 10)),
        )

        self.safety = dict(DEFAULT_SAFETY)
        self.safety.update(meta.get("safety") or {})
        self.safety.update(safety or {})
        self.policy_dt_s = float(meta.get("policy_dt_s", 0.1))

        self.fallback = fallback
        self.min_speed_mps = float(min_speed_mps)
        self.decay_tau_s = float(decay_tau_s)
        self._clock = clock

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Clear history, origin, velocity estimate and last action."""
        self._obs.reset()
        self._lat0 = None
        self._lon0 = None
        self._u = 0.0
        self._v = 0.0
        self._last_action = np.zeros(3, dtype=np.float64)
        self._last_t = None
        self.status = {"source": "init", "fault": None, "action": (0.0, 0.0, 0.0)}

    # ------------------------------------------------------------------- step
    def step(self, estimator_state, goal):
        """One policy step.

        Args:
            estimator_state: dict per CONTRACT §4.8 (`lat`, `lon`, `heading_deg`,
                `yaw_rate_deg_s`, `speed_mps`, `course_deg`, `gps_valid`,
                `imu_valid`, `mag_valid`, optional `sensor_ok`, `t`).
            goal: dict per CONTRACT §4.8 (`lat`, `lon`, optional `heading_deg`,
                `mode` = "waypoint" | "station_keeping", optional `next_lat`,
                `next_lon`, `next2_lat`, `next2_lon`).

        Returns:
            (surge, sway, yaw) floats in [-1, 1].
        """
        now = self._clock()
        prev_t = self._last_t
        dt_s = 0.0 if prev_t is None else now - prev_t
        self._last_t = now

        fault = self._check(estimator_state, now, prev_t)
        if fault is None and self._lat0 is None:
            self._lat0 = float(estimator_state["lat"])
            self._lon0 = float(estimator_state["lon"])

        if fault is None:
            east, north = latlon_to_enu(estimator_state["lat"], estimator_state["lon"],
                                        self._lat0, self._lon0)
            dist = float(np.hypot(east, north))
            if dist > float(self.safety["geofence_m"]):
                self._obs.reset()
                return self._emit(np.zeros(3), now, prev_t, "geofence", None)

            action, src, fault = self._policy(east, north, estimator_state, goal, dt_s)
        else:
            action, src = self._fallback_output(estimator_state, goal), "fallback"

        if src != "policy":
            # stale ring buffer must not replay into the first post-fault obs
            self._obs.reset()

        return self._emit(action, now, prev_t, src, fault)

    # --------------------------------------------------------------- internals
    def _check(self, est, now, prev_t):
        """Return a fault string or None."""
        wd = float(self.safety["watchdog_s"])
        t = est.get("t")
        if t is not None and now - float(t) > wd:
            return "watchdog"          # estimator data older than the watchdog
        if prev_t is not None and now - prev_t > wd:
            return "watchdog"          # caller stopped stepping
        if not bool(est.get("gps_valid", False)):
            return "gps"
        sensor_ok = est.get("sensor_ok")
        if sensor_ok is None:          # §4.4 semantics, derived when not supplied
            sensor_ok = (bool(est.get("imu_valid", True))
                         and bool(est.get("mag_valid", True)))
        if not bool(sensor_ok):
            return "sensor"
        if est.get("lat") is None or est.get("lon") is None:
            return "gps"
        return None

    def _policy(self, east, north, est, goal, dt_s):
        heading_deg = float(est.get("heading_deg", 0.0))
        psi = float(heading_deg_to_psi(heading_deg))

        # message-based decay: on a long gap decay hard instead of holding a stale (u,v)
        dt_est = float(np.clip(dt_s if dt_s > 0.0 else self.policy_dt_s, 1e-3, 0.5))
        self._u, self._v = estimate_body_velocity(
            float(est.get("speed_mps", 0.0)), float(est.get("course_deg", heading_deg)),
            heading_deg, dt_est, self._u, self._v, bool(est.get("gps_valid", False)),
            min_speed_mps=self.min_speed_mps, decay_tau_s=self.decay_tau_s)

        mode = goal.get("mode", "waypoint")
        mode_flag = _MODES.get(mode, float(mode) if isinstance(mode, (int, float)) else 0.0)
        hdg = goal.get("heading_deg")
        psi_goal = psi if hdg is None else float(heading_deg_to_psi(float(hdg)))
        g_e, g_n = latlon_to_enu(goal["lat"], goal["lon"], self._lat0, self._lon0)

        est_vec = np.array([[east, north, psi, self._u, self._v,
                             np.deg2rad(float(est.get("yaw_rate_deg_s", 0.0)))]])
        goal_vec = np.array([[g_e, g_n, psi_goal, mode_flag]])
        wp1 = self._wp_enu(goal, "next_lat", "next_lon")
        wp2 = self._wp_enu(goal, "next2_lat", "next2_lon")

        obs = self._obs.build(est_vec, goal_vec, self._last_action, wp1, wp2)
        out = self._session.run([self._output_name],
                                {self._input_name: np.ascontiguousarray(obs)})[0]
        action = np.asarray(out, dtype=np.float64).reshape(-1)[:3]
        if action.size != 3 or not np.all(np.isfinite(action)):
            return np.zeros(3), "zero", "nan"
        return action, "policy", None

    def _wp_enu(self, goal, klat, klon):
        """Later waypoint as absolute ENU [1,2]; None when the navigator gave none."""
        if goal.get(klat) is None or goal.get(klon) is None:
            return None
        e, n = latlon_to_enu(goal[klat], goal[klon], self._lat0, self._lon0)
        return np.array([[float(e), float(n)]])

    def _fallback_output(self, est, goal):
        if self.fallback is None:
            return np.zeros(3)
        out = self.fallback(est, goal)
        if out is None:
            return np.zeros(3)
        a = np.asarray(out, dtype=np.float64).reshape(-1)[:3]
        if a.size != 3 or not np.all(np.isfinite(a)):
            return np.zeros(3)
        return a

    def _emit(self, action, now, prev_t, source, fault):
        """Clip, slew-limit and remember the applied command."""
        clip = float(self.safety["clip"])
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -clip, clip)
        dt = self.policy_dt_s if prev_t is None else max(now - prev_t, 0.0)
        max_step = float(self.safety["rate_limit_per_s"]) * (dt or self.policy_dt_s)
        a = self._last_action + np.clip(a - self._last_action, -max_step, max_step)
        self._last_action = a
        self.status = {"source": source, "fault": fault,
                       "action": (float(a[0]), float(a[1]), float(a[2]))}
        return self.status["action"]
