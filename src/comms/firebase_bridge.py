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
      control:   { controller, source, fault }   # active controller + mux state
      timestamp: <unix>

  React -> Firebase -> RPi (commands):
    /vehicles/{id}/commands/
      mode: "idle" | "waypoint" | "station_keep" | "manual"
      controller: "pid" | "ppo"   # selects the controller; absent -> "pid"
      waypoints: [{ lat, lon, name, heading_hold }]
      manual:    { surge, sway, yaw }
      heading:   <degrees>   # for station_keep
      timestamp: <unix>
      ack:       <auto-set by RPi after processing>
"""

import asyncio
import json
import threading
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
    controller: str = "pid"                 # "pid" | "ppo"; unknown -> "pid"
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
        controller = data.get("controller", "pid")
        if controller not in ("pid", "ppo"):
            log.warning("Unknown controller %r in command -> 'pid'", controller)
            controller = "pid"
        return cls(
            mode=data.get("mode", "idle"),
            controller=controller,
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
        self._last_command_ts = time.time()   # Ignore commands older than boot

        # Pending telemetry buffer (queued between send cycles)
        self._telemetry: Dict[str, Any] = {}
        self._telemetry_lock = threading.Lock()

        # Command handler callback (set by main.py)
        self._command_handler: Optional[Callable[[RemoteCommand], Awaitable[None]]] = None

        # Streaming listener state (set up in command_listener_loop)
        self._listener = None
        self._cmd_loop: Optional[asyncio.AbstractEventLoop] = None
        self._cmd_queue: Optional[asyncio.Queue] = None

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
        with self._telemetry_lock:
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

    def push_control(self, controller: str, source: str, fault: Optional[str] = None):
        """Queue active-controller telemetry (node telemetry/control). Least
        invasive: a new additive node under the existing telemetry/ path, no
        existing node touched. controller = selected (pid|ppo); source = the
        mux's last emitted source; fault = mux fault or None."""
        self.queue_telemetry("telemetry/control", {
            "controller": controller,
            "source": source,
            "fault": fault or "",
        })

    def push_anomaly(self, gps: bool, imu: bool, mag: bool, score: float = 0.0):
        self.queue_telemetry("telemetry/anomaly", {
            "gps": gps, "imu": imu, "mag": mag, "score": round(score, 4),
        })

    def push_ml(self, payload: dict):
        """Queue ML layer telemetry (node telemetry/ml). Aditif — tidak ubah
        perilaku push lain. Dikirim per AGG_INTERVAL (harian), bukan per sampel.
        Lihat src/ml/SENSOR_CONTRACT.md untuk skema node."""
        self.queue_telemetry("telemetry/ml", payload)

    def _send_telemetry(self):
        """Batch-send all queued telemetry to Firebase.

        Swaps the buffer under a lock so the control-loop thread can keep
        queueing while this runs in the executor (no dict-mutation race).
        """
        if not self._enabled or self._ref is None:
            return

        now = time.time()
        if (now - self._last_telemetry_time) < (1.0 / self._log_rate):
            return

        with self._telemetry_lock:
            if not self._telemetry:
                return
            pending, self._telemetry = self._telemetry, {}

        try:
            update = {}
            for path, data in pending.items():
                for key, value in data.items():
                    update[f"{path}/{key}"] = value
            update["telemetry/timestamp"] = now

            self._ref.update(update)
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

            return cmd  # Dispatched in the async loop; ack written after it runs

        except Exception as e:
            log.error("Firebase command listen error: %s", e)
            time.sleep(2.0)  # Back off so a Firebase outage isn't hammered
            return None

    def _start_listen(self) -> bool:
        """Subscribe to the commands node (streaming push, no polling).

        firebase-admin runs the callback on its own thread and reconnects
        automatically, so every dashboard write is delivered — including a
        short-lived one that the next write overwrites a few ms later.
        """
        if not self._enabled or self._ref is None:
            return False
        try:
            self._listener = self._ref.child("commands").listen(
                self._on_command_event, self._on_command_error
            )
            log.info("Firebase command listener subscribed (streaming)")
            return True
        except Exception as e:
            log.error("Firebase listen failed: %s", e)
            return False

    def _on_command_event(self, event):
        """Runs on the firebase-admin stream thread (never blocks)."""
        try:
            data = event.data
            if not isinstance(data, dict):
                return  # node deleted

            cmd_ts = float(data.get("timestamp", 0))
            if cmd_ts <= self._last_command_ts:
                return  # already processed (also filters our own ack writes)

            self._last_command_ts = cmd_ts
            cmd = RemoteCommand.from_dict(data)
            log.info(
                "Command received: mode=%s, waypoints=%d, manual=[%.2f,%.2f,%.2f]",
                cmd.mode, len(cmd.waypoints),
                cmd.manual_surge, cmd.manual_sway, cmd.manual_yaw,
            )

            if self._cmd_loop is not None and self._cmd_queue is not None:
                self._cmd_loop.call_soon_threadsafe(self._cmd_queue.put_nowait, cmd)
        except Exception as e:
            log.error("Command event error: %s", e)

    def _on_command_error(self, exc: Exception):
        log.error("Firebase command stream error: %s", exc)

    def _ack_command(self, cmd: RemoteCommand, error: Optional[str] = None):
        """Write ack (or ack_error) only after the handler has actually run."""
        if not self._enabled or self._ref is None:
            return
        try:
            payload = {"ack": time.time(), "ack_mode": cmd.mode}
            if error:
                payload["ack_error"] = error[:200]
            self._ref.child("commands").update(payload)
        except Exception as e:
            log.error("Firebase ack write failed: %s", e)

    # ------------------------------------------------------------------
    # ASYNC LOOPS
    # ------------------------------------------------------------------
    async def telemetry_loop(self):
        """Async loop: pushes queued telemetry at configured rate."""
        if not self._running:
            self._running = True
        interval = 1.0 / self._log_rate
        log.info("Firebase telemetry loop started at %dHz", self._log_rate)

        while self._running:
            await asyncio.get_running_loop().run_in_executor(None, self._send_telemetry)
            await asyncio.sleep(interval)

    async def command_listener_loop(self):
        """Async loop: dispatch commands pushed by the Firebase stream listener.

        Polling is only a fallback — a 0.5s snapshot poll can miss a write that
        is overwritten before the next read, which is what made short joystick
        inputs get dropped.
        """
        if not self._running:
            self._running = True

        self._cmd_loop = asyncio.get_running_loop()
        self._cmd_queue = asyncio.Queue()

        streaming = await self._cmd_loop.run_in_executor(None, self._start_listen)
        if not streaming:
            log.warning("Streaming unavailable — falling back to 0.5s command polling")
            await self._poll_command_loop()
            return

        while self._running:
            try:
                cmd = await asyncio.wait_for(self._cmd_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._dispatch_command(cmd)

    async def _poll_command_loop(self):
        """Fallback: 0.5s snapshot polling (can drop overwritten writes)."""
        poll_interval = 0.5
        log.info("Firebase command polling started (%.1fs)", poll_interval)
        while self._running:
            cmd = await asyncio.get_running_loop().run_in_executor(
                None, self._listen_for_commands
            )
            if cmd is not None:
                await self._dispatch_command(cmd)
            await asyncio.sleep(poll_interval)

    async def _dispatch_command(self, cmd: RemoteCommand):
        if self._command_handler is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await self._command_handler(cmd)
        except Exception as e:
            log.error("Command handler error: %s", e)
            await loop.run_in_executor(None, self._ack_command, cmd, str(e))
        else:
            await loop.run_in_executor(None, self._ack_command, cmd)

    def stop(self):
        self._running = False
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception as e:
                log.error("Firebase listener close failed: %s", e)
            self._listener = None
        log.info("Firebase bridge stopping")

    # ------------------------------------------------------------------
    # Convenience: one-shot telemetry flush
    # ------------------------------------------------------------------
    def flush_now(self):
        """Force-send all queued telemetry immediately (blocking)."""
        self._last_telemetry_time = 0  # Bypass rate limit
        self._send_telemetry()
