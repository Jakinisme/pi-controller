"""
Sensor Fusion: Complementary filter combining GPS, MPU6050, and HMC5883L.

Heading strategy:
  - Magnetometer = primary absolute heading reference
  - Gyro Z (yaw rate) = smooth short-term changes between mag readings
  - GPS course = corrects long-term drift when moving at sufficient speed

Position strategy:
  - GPS = absolute position (noisy, low rate)
  - Low-pass filter smooths GPS jumps
"""

import asyncio
import time
import math
from dataclasses import dataclass
from typing import Optional

from src.sensors.gps import GPSReader
from src.sensors.imu import MPU6050Reader
from src.sensors.magnetometer import HMC5883LReader
from src.config import (
    FUSION_UPDATE_RATE_HZ,
    FUSION_GYRO_WEIGHT,
    FUSION_MAG_WEIGHT,
    FUSION_GPS_POSITION_ALPHA,
)
from src.utils.logger import setup_logger

log = setup_logger("fusion")


@dataclass
class FusedState:
    """Fused vehicle state from all sensors."""
    latitude: float = 0.0
    longitude: float = 0.0
    heading: float = 0.0         # degrees [0, 360), fused
    speed: float = 0.0           # m/s
    roll: float = 0.0            # degrees
    pitch: float = 0.0           # degrees
    yaw_rate: float = 0.0       # deg/s (from gyro, for derivative control)
    timestamp: float = 0.0
    gps_valid: bool = False
    imu_valid: bool = False
    mag_valid: bool = False


class SensorFusion:
    """Complementary filter fusing GPS + IMU + Magnetometer."""

    # Minimum GPS speed (m/s) to trust GPS heading
    GPS_HEADING_MIN_SPEED = 0.3

    def __init__(
        self,
        gps: GPSReader,
        imu: MPU6050Reader,
        mag: HMC5883LReader,
        update_rate_hz: int = FUSION_UPDATE_RATE_HZ,
        gyro_weight: float = FUSION_GYRO_WEIGHT,
        mag_weight: float = FUSION_MAG_WEIGHT,
        gps_alpha: float = FUSION_GPS_POSITION_ALPHA,
    ):
        self.gps = gps
        self.imu = imu
        self.mag = mag
        self._rate = update_rate_hz
        self._gyro_weight = gyro_weight
        self._mag_weight = mag_weight
        self._gps_alpha = gps_alpha

        self.state = FusedState()
        self._running = False
        self._last_update_time = 0.0
        self._gyro_heading = 0.0  # Integrated gyro heading

    def update(self) -> FusedState:
        """Run one fusion update cycle. Call at FUSION_UPDATE_RATE_HZ."""
        now = time.time()
        dt = now - self._last_update_time if self._last_update_time > 0 else 1.0 / self._rate
        self._last_update_time = now
        dt = min(dt, 0.5)  # Clamp dt to avoid huge jumps

        # --- Heading fusion ---
        imu_ok = self.imu.data.valid
        mag_ok = self.mag.data.valid
        gps_ok = self.gps.has_fix and not self.gps.is_stale

        if mag_ok and imu_ok:
            # Gyro integration: heading += yaw_rate * dt
            gyro_delta = self.imu.data.gyro_z * dt
            self._gyro_heading = (self._gyro_heading + gyro_delta) % 360

            # Complementary filter: blend gyro-propagated heading with magnetometer
            fused_heading = (
                self._gyro_weight * self._gyro_heading
                + self._mag_weight * self.mag.data.heading
            )

            # GPS course correction when moving fast enough
            if gps_ok and self.gps.data.speed_over_ground > self.GPS_HEADING_MIN_SPEED:
                gps_heading = self.gps.data.true_course
                # Small correction weight for GPS heading
                fused_heading = 0.95 * fused_heading + 0.05 * gps_heading

            self.state.heading = fused_heading % 360
            # Sync gyro heading to fused for next iteration
            self._gyro_heading = self.state.heading

        elif mag_ok:
            # No gyro, use magnetometer directly
            self.state.heading = self.mag.data.heading
            self._gyro_heading = self.mag.data.heading

        elif imu_ok:
            # No magnetometer, propagate from gyro (will drift!)
            gyro_delta = self.imu.data.gyro_z * dt
            self._gyro_heading = (self._gyro_heading + gyro_delta) % 360
            self.state.heading = self._gyro_heading
            log.warning("Heading from gyro only - will drift!")

        # --- Position fusion ---
        if gps_ok:
            alpha = self._gps_alpha
            if self.state.latitude == 0.0:
                # First fix, just set directly
                self.state.latitude = self.gps.data.latitude
                self.state.longitude = self.gps.data.longitude
            else:
                self.state.latitude = alpha * self.gps.data.latitude + (1 - alpha) * self.state.latitude
                self.state.longitude = alpha * self.gps.data.longitude + (1 - alpha) * self.state.longitude
            self.state.speed = self.gps.data.speed_over_ground
            self.state.gps_valid = True
        else:
            self.state.gps_valid = False

        # --- Orientation from IMU ---
        if imu_ok:
            self.state.roll = self.imu.data.roll
            self.state.pitch = self.imu.data.pitch
            self.state.yaw_rate = self.imu.data.gyro_z
            self.state.imu_valid = True

        self.state.mag_valid = mag_ok
        self.state.timestamp = now
        return self.state

    async def update_loop(self):
        """Async loop running fusion at configured rate."""
        self._running = True
        interval = 1.0 / self._rate

        log.info("Sensor fusion started at %dHz", self._rate)
        while self._running:
            t0 = time.monotonic()
            try:
                self.update()
            except Exception as e:
                log.error("Fusion update error: %s", e)

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def stop(self):
        self._running = False
        log.info("Sensor fusion stopping")

    def get_state(self) -> FusedState:
        """Return the current fused state (thread-safe read)."""
        return self.state

