"""
ASV pi-control Main Entry Point.

Orchestrates all subsystems via asyncio:
  - Sensor reading (GPS, IMU, Magnetometer)
  - Sensor fusion
  - Navigation / PID control
  - ESP32 communication
  - ML inference
  - Firebase bridge (bidirectional: push telemetry + subscribe commands)
  - Camera stream (RTSP -> HLS for React dashboard)

Architecture:
  RPi  <-- Firebase RTDB -->  React Dashboard (separate project)
  RPi  -- HLS stream URL -->  React Dashboard (video via HLS.js)

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
    ML_ANOMALY_ENABLED,
    ML_PREDICTION_ENABLED,
    ML_INFERENCE_RATE_HZ,
    FIREBASE_VEHICLE_ID,
    CAMERA_ENABLED,
)
from src.sensors.gps import GPSReader
from src.sensors.imu import MPU6050Reader
from src.sensors.magnetometer import HMC5883LReader
from src.sensors.fusion import SensorFusion
from src.control.navigation import WaypointNavigator, Waypoint, NavigationState
from src.control.thrust import ThrustComputer
from src.comms.esp32_link import ESP32Link
from src.comms.firebase_bridge import FirebaseBridge, RemoteCommand
# pyrefly: ignore [missing-import]
from src.ml.anomaly.py import AnomalyDetector
# pyrefly: ignore [missing-import]
from src.ml.prediction.py import DriftPredictor
from src.camera.rtsp_streamer import RTSPStreamer
from src.utils.logger import setup_logger

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

        # Communication
        self.esp32 = ESP32Link()
        self.firebase = FirebaseBridge(vehicle_id=FIREBASE_VEHICLE_ID)

        # Camera
        self.camera = RTSPStreamer()

        # ML
        self.anomaly_detector = AnomalyDetector()
        self.drift_predictor = DriftPredictor()

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

        if cmd.mode == "idle":
            self.navigator.stop()
            self.thrust.reset_all()

        elif cmd.mode == "station_keep":
            state = self.fusion.get_state()
            hold_heading = cmd.heading if cmd.heading is not None else state.heading
            self.navigator.start_station_keeping(
                state.latitude, state.longitude, hold_heading
            )

        elif cmd.mode == "waypoint":
            wp_list = []
            for wp_data in cmd.waypoints:
                wp_list.append(Waypoint(
                    latitude=float(wp_data.get("lat", 0)),
                    longitude=float(wp_data.get("lon", 0)),
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
                thrust_cmd = self.thrust.compute(nav_output, state)

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
                )

            except Exception as e:
                log.error("Control loop error: %s", e)

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    # ------------------------------------------------------------------
    # ML LOOP  (~1Hz)
    # ------------------------------------------------------------------
    async def ml_loop(self):
        """ML inference loop (~1Hz, non-blocking)."""
        if not ML_ANOMALY_ENABLED and not ML_PREDICTION_ENABLED:
            log.info("ML disabled")
            return

        interval = 1.0 / ML_INFERENCE_RATE_HZ
        log.info("ML loop started at %dHz", ML_INFERENCE_RATE_HZ)

        while self._running:
            t0 = time.monotonic()

            try:
                state = self.fusion.get_state()
                cmd = self.thrust.command

                if ML_ANOMALY_ENABLED:
                    result = self.anomaly_detector.detect(
                        gps_speed=state.speed,
                        lat=state.latitude,
                        lon=state.longitude,
                        gyro_z=state.yaw_rate,
                        accel_x=self.imu.data.accel_x,
                        accel_y=self.imu.data.accel_y,
                        accel_z=self.imu.data.accel_z,
                        mag_heading=self.mag.data.heading,
                    )
                    self.firebase.push_anomaly(
                        result.gps_anomaly,
                        result.imu_anomaly,
                        result.mag_anomaly,
                        score=result.anomaly_score,
                    )

                    # Auto-fit model when enough data collected
                    if (not self.anomaly_detector._is_fitted
                            and len(self.anomaly_detector._history) >= 200):
                        self.anomaly_detector.fit()

                if ML_PREDICTION_ENABLED:
                    self.drift_predictor.predict(
                        heading=state.heading,
                        speed=state.speed,
                        surge=cmd.surge,
                        sway=cmd.sway,
                        yaw=cmd.yaw,
                    )

            except Exception as e:
                log.error("ML loop error: %s", e)

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    # ------------------------------------------------------------------
    # CAMERA HEALTH MONITOR  (~0.1Hz)
    # ------------------------------------------------------------------
    async def camera_health_loop(self):
        """Periodically check camera health and push stream URL to Firebase."""
        while self._running:
            if self.camera.is_running:
                healthy = await self.camera.health_check()
                if not healthy:
                    log.warning("Camera stream unhealthy, restarting...")
                    await self.camera.stop()
                    rpi_ip = get_local_ip()
                    await self.camera.start(rpi_ip)
                    self.firebase.push_camera_url(self.camera.stream_url)
            await asyncio.sleep(10)

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

        # Start camera stream
        if CAMERA_ENABLED:
            started = await self.camera.start(rpi_ip)
            if started:
                self.firebase.push_camera_url(self.camera.stream_url)
                log.info("Camera stream URL: %s", self.camera.stream_url)

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

        if CAMERA_ENABLED:
            tasks.append(
                asyncio.create_task(self.camera_health_loop(), name="camera_health")
            )

        log.info("All subsystems started (%d tasks)", len(tasks))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("Tasks cancelled")
        except Exception as e:
            log.error("Unexpected error in main: %s", e)

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
        await self.camera.stop()

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
