"""
ASV pi-control Main Entry Point.

Orchestrates all subsystems via asyncio:
  - Sensor reading (GPS, IMU, Magnetometer)
  - Sensor fusion
  - Navigation / PID control
  - ESP32 communication (UART serial)
  - ML inference
  - Firebase bridge (bidirectional: push telemetry + subscribe commands)

Architecture:
  RPi  <-- Firebase RTDB -->  React Dashboard (separate project)
  RPi  -- UART serial ------>  ESP32 (motor controller)

Usage:
    python -m src.main
"""

import asyncio
import signal
import socket
import sys
import time
import os            

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FUSION_UPDATE_RATE_HZ,
    FIREBASE_VEHICLE_ID,
    RL_LOG_DIR,
)
from src.sensors.gps import GPSReader
from src.sensors.imu import MPU6050Reader
from src.sensors.magnetometer import HMC5883LReader
from src.sensors.fusion import SensorFusion
from src.control.navigation import WaypointNavigator, Waypoint, NavigationState
from src.control.thrust import ThrustComputer
from src.comms.esp32_link import ESP32Link
from src.comms.firebase_bridge import FirebaseBridge, RemoteCommand
from src.control.controller_mux import ControllerMux
from src.rl.adapter import RLAdapter
from src.rl.logger import RLLogger
# ML modules disabled — uncomment when available
# from src.ml.anomaly import AnomalyDetector
# from src.ml.prediction import DriftPredictor
from src.utils.logger import setup_logger
from src.ml.integration import MLFanout
from src.ml import config_ml as cfg_ml

log = setup_logger("main")


def get_local_ip() -> str:
    """Get the RPi's IP on the local network (MiFi)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


class ASVSystem:
    """Top-level system coordinator."""

    def __init__(self):
        # Sensors
        self.gps = GPSReader()
        self.imu = MPU6050Reader()
        self.mag = HMC5883LReader()

        # Fusion
        self.fusion = SensorFusion(self.gps, self.imu, self.mag)

        # Navigation & Control
        self.navigator = WaypointNavigator()
        self.thrust = ThrustComputer()

        # RL policy adapter (fail-soft: available=False + a reason when
        # onnxruntime or the model is missing) + the safety mux that owns
        # the PID/PPO decision. self.thrust stays: manual/reset use it.
        #
        # One shared clock feeds both: the mux stamps est["t"] from it and
        # RLController runs its own watchdog against it, so they must be the
        # same source or every policy step would look stale.
        self._rl_clock = time.monotonic
        self.rl_adapter = RLAdapter.from_config(clock=self._rl_clock)
        self.rl_logger = RLLogger(RL_LOG_DIR)
        self.controller = ControllerMux(
            self.thrust, self.rl_adapter, self.navigator, self.gps,
            clock=self._rl_clock, rl_logger=self.rl_logger,
        )

        # Communication
        self.esp32 = ESP32Link()
        self.firebase = FirebaseBridge(vehicle_id=FIREBASE_VEHICLE_ID)

        # ML (disabled)
        self.anomaly_detector = None
        self.drift_predictor = None

        # ML fan-out layer (shadow mode, scoring OFF). Aditif — tidak
        # kirim perintah motor. Lihat src/ml/SENSOR_CONTRACT.md.
        self.ml_fanout = None

        # Control loop rate
        self._control_rate_hz = FUSION_UPDATE_RATE_HZ
        self._running = False

        # Register Firebase command handler
        self.firebase.on_command(self._handle_remote_command)

    # ------------------------------------------------------------------
    # REMOTE COMMAND HANDLER  (React Dashboard -> Firebase -> RPi)
    # ------------------------------------------------------------------
    async def _handle_remote_command(self, cmd: RemoteCommand):
        """Process a command received from the React dashboard via Firebase."""
        log.info("Processing remote command: mode=%s", cmd.mode)
        # Takes effect within one control tick; absent field -> "pid"
        # (RemoteCommand default), every branch below unchanged.
        self.controller.set_controller_request(cmd.controller)

        if cmd.mode == "idle":
            self.navigator.stop()
            self.thrust.reset_all()

        elif cmd.mode == "station_keep":
            state = self.fusion.get_state()
            if not state.gps_valid:
                log.warning("Station-keep rejected: no GPS fix")
                return
            hold_heading = cmd.heading if cmd.heading is not None else state.heading
            self.navigator.start_station_keeping(
                state.latitude, state.longitude, hold_heading
            )

        elif cmd.mode == "waypoint":
            wp_list = []
            for wp_data in cmd.waypoints:
                try:
                    lat = float(str(wp_data.get("lat")))
                    lon = float(str(wp_data.get("lon")))
                except (TypeError, ValueError):
                    log.warning("Waypoint rejected: non-numeric lat/lon in %s", wp_data)
                    return
                if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0) \
                        or (lat == 0.0 and lon == 0.0):
                    log.warning("Waypoint rejected: out-of-range (%s, %s)", lat, lon)
                    return
                wp_list.append(Waypoint(
                    latitude=lat,
                    longitude=lon,
                    name=str(wp_data.get("name", "")),
                    heading_hold=wp_data.get("heading_hold"),
                ))
            if wp_list:
                self.navigator.start_waypoint_nav(wp_list)
            else:
                log.warning("Waypoint command with empty waypoint list")

        elif cmd.mode == "manual":
            self.navigator.set_manual()
            self.thrust.set_manual_thrust(
                cmd.manual_surge, cmd.manual_sway, cmd.manual_yaw
            )

        else:
            log.warning("Unknown command mode: %s", cmd.mode)

    # ------------------------------------------------------------------
    # CONTROL LOOP  (~20Hz)
    # ------------------------------------------------------------------
    async def control_loop(self):
        """Main control loop: navigation -> PID -> thrust -> ESP32.
        Runs at FUSION_UPDATE_RATE_HZ (~20Hz).
        """
        interval = 1.0 / self._control_rate_hz
        log.info("Control loop started at %dHz", self._control_rate_hz)
        _last_status_log = 0.0       # periodic status print
        _STATUS_LOG_INTERVAL = 5.0   # seconds

        while self._running:
            t0 = time.monotonic()

            try:
                # 1. Get fused state
                state = self.fusion.get_state()

                # 2. Run navigation (waypoint or station-keeping)
                nav_output = self.navigator.update(
                    lat=state.latitude,
                    lon=state.longitude,
                    heading=state.heading,
                    gps_valid=state.gps_valid,
                )

                # 3. Compute thrust via PID controllers
                thrust_cmd = self.controller.compute(nav_output, state)

                # 4. Send to ESP32
                self.esp32.send_thrust(thrust_cmd)

                # 5. Queue telemetry for Firebase
                self.firebase.push_position(
                    state.latitude, state.longitude,
                    state.heading, state.speed,
                    altitude=self.gps.data.altitude,
                )
                self.firebase.push_thrust(
                    thrust_cmd.surge, thrust_cmd.sway, thrust_cmd.yaw,
                )
                self.firebase.push_sensors(
                    state.gps_valid, state.imu_valid, state.mag_valid,
                    num_satellites=self.gps.data.num_satellites,
                )
                self.firebase.push_nav(
                    self.navigator.state.name,
                    self.navigator.current_wp_index,
                    len(self.navigator.waypoints),
                    distance_to_wp=self.navigator.distance_to_waypoint,
                )
                self.firebase.push_control(
                    self.controller.selected, self.controller.source,
                    self.controller.fault,
                )

                # --- Periodic status log (every 5s) ---
                now_mono = time.monotonic()
                if now_mono - _last_status_log >= _STATUS_LOG_INTERVAL:
                    _last_status_log = now_mono
                    gd = self.gps.data
                    log.info(
                        "STATUS | GPS fix=%s sats=%d | "
                        "pos=(%.6f, %.6f) spd=%.2f m/s crs=%.1f° | "
                        "heading=%.1f° | "
                        "thrust S=%.3f W=%.3f Y=%.3f | nav=%s",
                        gd.valid, gd.num_satellites,
                        state.latitude, state.longitude,
                        state.speed, gd.true_course,
                        state.heading,
                        thrust_cmd.surge, thrust_cmd.sway, thrust_cmd.yaw,
                        self.navigator.state.name,
                    )

            except Exception as e:
                log.error("Control loop error: %s", e)

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    # ------------------------------------------------------------------
    # ML LOOP  (fan-out + aggregate per AGG_INTERVAL)
    # ------------------------------------------------------------------
    async def ml_loop(self):
        """ML shadow layer: fan-out IMU/GPS read-only, aggregate per
        AGG_INTERVAL, log + push telemetry/ml. Scoring OFF (aturan 6.6).
        Tidak kirim perintah motor."""
        self.ml_fanout = MLFanout(self.imu, self.gps, self.firebase)
        log.info("ML fan-out started (shadow, scoring=OFF)")
        await self.ml_fanout.ml_loop()

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    async def run(self):
        """Start all subsystems."""
        self._running = True
        log.info("=" * 60)
        log.info("ASV pi-control starting...")
        log.info("=" * 60)

        rpi_ip = get_local_ip()
        log.info("RPi IP: %s", rpi_ip)

        # Initialize hardware
        if not self.imu.initialize():
            log.warning("IMU init failed - will retry in loop")
        else:
            self.imu.calibrate_gyro()

        if not self.mag.initialize():
            log.warning("Magnetometer init failed - will retry in loop")

        if not self.esp32.open():
            log.warning("ESP32 link failed - will retry")

        # Build task list
        tasks = [
            asyncio.create_task(self.gps.read_loop(), name="gps"),
            asyncio.create_task(self.imu.read_loop(), name="imu"),
            asyncio.create_task(self.mag.read_loop(), name="mag"),
            asyncio.create_task(self.fusion.update_loop(), name="fusion"),
            asyncio.create_task(self.control_loop(), name="control"),
            asyncio.create_task(self.esp32.heartbeat_loop(), name="esp32_hb"),
            asyncio.create_task(self.ml_loop(), name="ml"),
            # Firebase bidirectional: push telemetry + listen for commands
            asyncio.create_task(self.firebase.telemetry_loop(), name="fb_telemetry"),
            asyncio.create_task(self.firebase.command_listener_loop(), name="fb_commands"),
        ]

        log.info("All subsystems started (%d tasks)", len(tasks))

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    log.error(
                        "Task %s crashed: %r — shutting down",
                        task.get_name(), exc,
                    )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            log.info("Tasks cancelled")
        finally:
            # Always run cleanup: zero thrust, close serial/GPIO, stop loops.
            await self.shutdown()

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------
    async def shutdown(self):
        """Graceful shutdown of all subsystems."""
        log.info("Shutting down...")
        self._running = False

        # Stop all loops
        self.gps.stop()
        self.imu.stop()
        self.mag.stop()
        self.fusion.stop()
        self.esp32.stop()
        self.firebase.stop()
        if self.ml_fanout:
            self.ml_fanout.stop()

        # Flush the RL black-box log (sysid/replay input) before hardware closes
        try:
            self.rl_logger.close()
        except Exception as e:
            log.warning("RL logger close failed: %s", e)

        # Close hardware
        self.gps.close()
        self.imu.close()
        self.mag.close()
        self.esp32.close()

        log.info("Shutdown complete")


def main():
    """Entry point."""
    system = ASVSystem()

    # Handle signals for graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler():
        log.info("Signal received, shutting down...")
        loop.create_task(system.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(system.run())
    except KeyboardInterrupt:
        loop.run_until_complete(system.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
