"""
Firebase Realtime Database Bridge — Bidirectional Communication.

RPi <-- Firebase RTDB --> React Dashboard

DATA FLOW:
  RPi -> Firebase (telemetry):
    /vehicles/{id}/telemetry/
      position:  { lat, lon, heading, speed, altitude }
      thrust:    { surge, sway, yaw }
      sensors:   { gps_valid, imu_valid, mag_valid, num_satellites }
      anomaly:   { gps, imu, mag, score }
      nav:       { mode, waypoint_index, total_waypoints, distance_to_wp }
      camera:    { stream_url }
      timestamp: <unix>

  React -> Firebase -> RPi (commands):
    /vehicles/{id}/commands/
      mode: "idle" | "waypoint" | "station_keep" | "manual"
      waypoints: [{ lat, lon, name, heading_hold }]
      manual:    { surge, sway, yaw }
      heading:   <degrees>   # for station_keep
      timestamp: <unix>
      ack:       <auto-set by RPi after processing>
"""

import asyncio
import json
import time
from typing import Optional, Callable, Awaitable, Any, Dict, List
from dataclasses import dataclass, field

from src.config import (
    FIREBASE_ENABLED,
    FIREBASE_CREDENTIALS_PATH,
    FIREBASE_DATABASE_URL,
    FIREBASE_LOG_RATE_HZ,
)
from src.utils.logger import setup_logger

log = setup_logger("firebase")

try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
except ImportError:
    firebase_admin = None
    log.warning("firebase-admin not installed; Firebase bridge disabled")


# ---------------------------------------------------------------------------
# Command dataclass (parsed from Firebase)
# ---------------------------------------------------------------------------
@dataclass
class RemoteCommand:
    """Command received from React dashboard via Firebase."""
    mode: str = "idle"                      # idle, waypoint, station_keep, manual
    waypoints: List[Dict[str, Any]] = field(default_factory=list)
    manual_surge: float = 0.0
    manual_sway: float = 0.0
    manual_yaw: float = 0.0
    heading: Optional[float] = None         # for station_keep
    timestamp: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RemoteCommand":
        manual = data.get("manual", {})
        wps = data.get("waypoints", [])
        return cls(
            mode=data.get("mode", "idle"),
            waypoints=wps if isinstance(wps, list) else list(wps.values()) if isinstance(wps, dict) else [],
            manual_surge=float(manual.get("surge", 0)),
            manual_sway=float(manual.get("sway", 0)),
            manual_yaw=float(manual.get("yaw", 0)),
            heading=data.get("heading"),
            timestamp=float(data.get("timestamp", 0)),
            raw=data,
        )


# ---------------------------------------------------------------------------
# Firebase Bridge
# ---------------------------------------------------------------------------
class FirebaseBridge:
    """Bidirectional Firebase RTDB bridge.

    - Pushes vehicle telemetry to Firebase at ~1Hz
    - Subscribes to command path and dispatches callbacks to the control system
    """

    def __init__(
        self,
        vehicle_id: str = "asv-001",
        enabled: bool = FIREBASE_ENABLED,
        credentials_path: str = FIREBASE_CREDENTIALS_PATH,
        database_url: str = FIREBASE_DATABASE_URL,
        log_rate_hz: int = FIREBASE_LOG_RATE_HZ,
    ):
        self._vehicle_id = vehicle_id
        self._enabled = enabled
        self._db_url = database_url
        self._log_rate = log_rate_hz
        self._running = False
        self._ref = None              # Root ref: /vehicles/{id}
        self._last_telemetry_time = 0.0
        self._last_command_ts = 0.0   # Deduplicate repeated commands

        # Pending telemetry buffer (queued between send cycles)
        self._telemetry: Dict[str, Any] = {}

        # Command handler callback (set by main.py)
        self._command_handler: Optional[Callable[[RemoteCommand], Awaitable[None]]] = None

        if not enabled:
            log.info("Firebase bridge disabled")
            return

        self._init_firebase(credentials_path)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_firebase(self, cred_path: str):
        if firebase_admin is None:
            log.error("firebase-admin not installed")
            self._enabled = False
            return
        try:
            cred = credentials.Certificate(cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {"databaseURL": self._db_url})
            self._ref = firebase_db.reference(f"/vehicles/{self._vehicle_id}")
            log.info("Firebase bridge initialized: %s/%s", self._db_url, self._vehicle_id)
        except Exception as e:
            log.error("Firebase init failed: %s", e)
            self._enabled = False

    def on_command(self, handler: Callable[[RemoteCommand], Awaitable[None]]):
        """Register an async callback for incoming commands from the dashboard."""
        self._command_handler = handler
        log.info("Command handler registered")

    # ------------------------------------------------------------------
    # TELEMETRY PUSH  (RPi -> Firebase)
    # ------------------------------------------------------------------
    def queue_telemetry(self, path: str, data: dict):
        """Queue telemetry data to be sent on the next push cycle."""
        self._telemetry[path] = data

    def push_position(self, lat: float, lon: float, heading: float, speed: float, altitude: float = 0.0):
        self.queue_telemetry("telemetry/position", {
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "heading": round(heading, 1),
            "speed": round(speed, 3),
            "altitude": round(altitude, 1),
        })

    def push_thrust(self, surge: float, sway: float, yaw: float):
        self.queue_telemetry("telemetry/thrust", {
            "surge": round(surge, 3),
            "sway": round(sway, 3),
            "yaw": round(yaw, 3),
        })

    def push_sensors(self, gps_valid: bool, imu_valid: bool, mag_valid: bool, num_satellites: int = 0):
        self.queue_telemetry("telemetry/sensors", {
            "gps_valid": gps_valid,
            "imu_valid": imu_valid,
            "mag_valid": mag_valid,
            "num_satellites": num_satellites,
        })

    def push_nav(self, mode: str, waypoint_index: int, total_waypoints: int, distance_to_wp: float = 0.0):
        self.queue_telemetry("telemetry/nav", {
            "mode": mode,
            "waypoint_index": waypoint_index,
            "total_waypoints": total_waypoints,
            "distance_to_wp": round(distance_to_wp, 2),
        })

    def push_anomaly(self, gps: bool, imu: bool, mag: bool, score: float = 0.0):
        self.queue_telemetry("telemetry/anomaly", {
            "gps": gps, "imu": imu, "mag": mag, "score": round(score, 4),
        })

    def push_camera_url(self, stream_url: str):
        """Push the RTSP/HLS stream URL so the React dashboard knows where to connect."""
        self.queue_telemetry("telemetry/camera", {"stream_url": stream_url})

    def _send_telemetry(self):
        """Batch-send all queued telemetry to Firebase."""
        if not self._enabled or self._ref is None:
            return
        if not self._telemetry:
            return

        now = time.time()
        if (now - self._last_telemetry_time) < (1.0 / self._log_rate):
            return

        try:
            update = {}
            for path, data in self._telemetry.items():
                for key, value in data.items():
                    update[f"{path}/{key}"] = value
            update["telemetry/timestamp"] = now

            self._ref.update(update)
            self._telemetry.clear()
            self._last_telemetry_time = now
        except Exception as e:
            log.error("Firebase telemetry push failed: %s", e)

    # ------------------------------------------------------------------
    # COMMAND SUBSCRIBE  (Firebase -> RPi)
    # ------------------------------------------------------------------
    def _listen_for_commands(self):
        """Poll the commands node for new data (blocking, run in executor)."""
        if not self._enabled or self._ref is None:
            time.sleep(0.5)
            return

        try:
            cmd_ref = self._ref.child("commands")
            data = cmd_ref.get()
            if not data or not isinstance(data, dict):
                return

            cmd_ts = float(data.get("timestamp", 0))
            if cmd_ts <= self._last_command_ts:
                return  # Already processed

            self._last_command_ts = cmd_ts
            cmd = RemoteCommand.from_dict(data)
            log.info(
                "Command received: mode=%s, waypoints=%d, manual=[%.2f,%.2f,%.2f]",
                cmd.mode, len(cmd.waypoints),
                cmd.manual_surge, cmd.manual_sway, cmd.manual_yaw,
            )

            # Acknowledge by writing back
            cmd_ref.update({"ack": time.time(), "ack_mode": cmd.mode})

            return cmd  # Will be dispatched in the async loop

        except Exception as e:
            log.error("Firebase command listen error: %s", e)
            return None

    # ------------------------------------------------------------------
    # ASYNC LOOPS
    # ------------------------------------------------------------------
    async def telemetry_loop(self):
        """Async loop: pushes queued telemetry at configured rate."""
        self._running = True
        interval = 1.0 / self._log_rate
        log.info("Firebase telemetry loop started at %dHz", self._log_rate)

        while self._running:
            await asyncio.get_event_loop().run_in_executor(None, self._send_telemetry)
            await asyncio.sleep(interval)

    async def command_listener_loop(self):
        """Async loop: polls Firebase for new commands and dispatches handler."""
        self._running = True
        poll_interval = 0.5  # Check every 500ms
        log.info("Firebase command listener started (poll %.1fs)", poll_interval)

        while self._running:
            cmd = await asyncio.get_event_loop().run_in_executor(
                None, self._listen_for_command
            )
            if cmd is not None and self._command_handler is not None:
                try:
                    await self._command_handler(cmd)
                except Exception as e:
                    log.error("Command handler error: %s", e)

            await asyncio.sleep(poll_interval)

    def stop(self):
        self._running = False
        log.info("Firebase bridge stopping")

    # ------------------------------------------------------------------
    # Convenience: one-shot telemetry flush
    # ------------------------------------------------------------------
    def flush_now(self):
        """Force-send all queued telemetry immediately (blocking)."""
        self._last_telemetry_time = 0  # Bypass rate limit
        self._send_telemetry()
