"""
MPU6050 Accelerometer + Gyroscope reader via I2C.
Provides roll, pitch, and angular rates. No magnetometer on this chip.
"""

import asyncio
import time
import math
import struct
from dataclasses import dataclass
from typing import Optional

from smbus2 import SMBus

from src.config import MPU6050_I2C_BUS, MPU6050_I2C_ADDR, MPU6050_SAMPLE_RATE_HZ
from src.utils.logger import setup_logger

log = setup_logger("imu")

# MPU6050 Register Map
MPU6050_REG_PWR_MGMT_1 = 0x6B
MPU6050_REG_PWR_MGMT_2 = 0x6C
MPU6050_REG_SMPLRT_DIV = 0x19
MPU6050_REG_CONFIG = 0x1A
MPU6050_REG_GYRO_CONFIG = 0x1B
MPU6050_REG_ACCEL_CONFIG = 0x1C
MPU6050_REG_ACCEL_XOUT_H = 0x3B
MPU6050_REG_TEMP_OUT_H = 0x41
MPU6050_REG_GYRO_XOUT_H = 0x43
MPU6050_REG_WHO_AM_I = 0x75

# Scale factors
ACCEL_SCALE = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}  # LSB/g
GYRO_SCALE = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}  # LSB/(deg/s)


@dataclass
class IMUData:
    """Container for MPU6050 sensor data."""
    accel_x: float = 0.0  # m/s^2
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0   # deg/s
    gyro_y: float = 0.0
    gyro_z: float = 0.0   # This is yaw rate (important for heading PID)
    roll: float = 0.0     # degrees (from accel)
    pitch: float = 0.0    # degrees (from accel)
    temperature: float = 0.0  # Celsius
    timestamp: float = 0.0
    valid: bool = False


class MPU6050Reader:
    """Reader for MPU6050 IMU via I2C."""

    def __init__(
        self,
        bus: int = MPU6050_I2C_BUS,
        addr: int = MPU6050_I2C_ADDR,
        accel_range: int = 2,
        gyro_range: int = 250,
        sample_rate_hz: int = MPU6050_SAMPLE_RATE_HZ,
    ):
        self._bus_id = bus
        self._addr = addr
        self._accel_range = accel_range
        self._gyro_range = gyro_range
        self._sample_rate = sample_rate_hz
        self._bus: Optional[SMBus] = None
        self.data = IMUData()
        self._running = False

        # Calibration offsets (calibrate on startup)
        self._gyro_offset_x = 0.0
        self._gyro_offset_y = 0.0
        self._gyro_offset_z = 0.0
        self._is_calibrated = False

    def initialize(self) -> bool:
        """Initialize MPU6050 registers. Returns True on success.

        Retries up to 3 times with increasing delay to handle transient
        I2C errors that can occur right after power-up.
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if self._bus is None:
                    self._bus = SMBus(self._bus_id)

                # Check WHO_AM_I register
                who = self._bus.read_byte_data(self._addr, MPU6050_REG_WHO_AM_I)
                if who not in (0x68, 0x98):  # 0x68 for genuine, 0x98 for clones
                    log.error("MPU6050 WHO_AM_I returned 0x%02X, expected 0x68", who)
                    return False

                # Wake up (clear sleep bit)
                self._bus.write_byte_data(self._addr, MPU6050_REG_PWR_MGMT_1, 0x00)
                time.sleep(0.1)

                # Set sample rate divider: 1kHz / (1 + SMPLRT_DIV)
                divider = max(0, min(255, int(1000 / self._sample_rate) - 1))
                self._bus.write_byte_data(self._addr, MPU6050_REG_SMPLRT_DIV, divider)

                # Set DLPF to ~44Hz bandwidth (good noise reduction)
                self._bus.write_byte_data(self._addr, MPU6050_REG_CONFIG, 0x03)

                # Set accelerometer range
                accel_bits = {2: 0x00, 4: 0x08, 8: 0x10, 16: 0x18}[self._accel_range]
                self._bus.write_byte_data(self._addr, MPU6050_REG_ACCEL_CONFIG, accel_bits)

                # Set gyroscope range
                gyro_bits = {250: 0x00, 500: 0x08, 1000: 0x10, 2000: 0x18}[self._gyro_range]
                self._bus.write_byte_data(self._addr, MPU6050_REG_GYRO_CONFIG, gyro_bits)

                log.info(
                    "MPU6050 initialized: accel=%dg, gyro=%ddeg/s, rate=%dHz",
                    self._accel_range, self._gyro_range, self._sample_rate,
                )
                return True

            except Exception as e:
                if attempt < max_retries:
                    delay = 0.2 * attempt  # 0.2s, 0.4s
                    log.warning(
                        "MPU6050 init attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt, max_retries, e, delay,
                    )
                    time.sleep(delay)
                else:
                    log.error("MPU6050 init failed after %d attempts: %s", max_retries, e)
                    return False

    def calibrate_gyro(self, samples: int = 200):
        """Calibrate gyroscope offsets. Keep the IMU still during calibration."""
        log.info("Calibrating MPU6050 gyro (%d samples)... Keep IMU still!", samples)
        gx_sum = gy_sum = gz_sum = 0.0
        success_count = 0

        for _ in range(samples):
            raw = self._read_raw_gyro()
            if raw:
                gx_sum += raw[0]
                gy_sum += raw[1]
                gz_sum += raw[2]
                success_count += 1
            time.sleep(0.01)

        if success_count == 0:
            log.warning("Gyro calibration failed: no successful reads")
            return

        self._gyro_offset_x = gx_sum / success_count
        self._gyro_offset_y = gy_sum / success_count
        self._gyro_offset_z = gz_sum / success_count
        self._is_calibrated = True
        log.info(
            "Gyro offsets: x=%.2f, y=%.2f, z=%.2f deg/s (%d/%d reads)",
            self._gyro_offset_x, self._gyro_offset_y, self._gyro_offset_z,
            success_count, samples,
        )

    def _read_raw_accel(self):
        """Read raw accelerometer values (returns g)."""
        try:
            raw = self._bus.read_i2c_block_data(self._addr, MPU6050_REG_ACCEL_XOUT_H, 6)
            scale = ACCEL_SCALE[self._accel_range]
            ax = struct.unpack(">h", bytes(raw[0:2]))[0] / scale
            ay = struct.unpack(">h", bytes(raw[2:4]))[0] / scale
            az = struct.unpack(">h", bytes(raw[4:6]))[0] / scale
            return ax, ay, az
        except Exception:
            return None

    def _read_raw_gyro(self):
        """Read raw gyroscope values (returns deg/s before offset)."""
        try:
            raw = self._bus.read_i2c_block_data(self._addr, MPU6050_REG_GYRO_XOUT_H, 6)
            scale = GYRO_SCALE[self._gyro_range]
            gx = struct.unpack(">h", bytes(raw[0:2]))[0] / scale
            gy = struct.unpack(">h", bytes(raw[2:4]))[0] / scale
            gz = struct.unpack(">h", bytes(raw[4:6]))[0] / scale
            return gx, gy, gz
        except Exception:
            return None

    def _read_temperature(self) -> float:
        """Read on-chip temperature sensor."""
        try:
            raw = self._bus.read_i2c_block_data(self._addr, MPU6050_REG_TEMP_OUT_H, 2)
            temp_raw = struct.unpack(">h", bytes(raw[0:2]))[0]
            return temp_raw / 340.0 + 36.53
        except Exception:
            return 0.0

    def read(self) -> bool:
        """Perform a single read cycle, updating self.data. Returns True on success."""
        accel = self._read_raw_accel()
        gyro = self._read_raw_gyro()
        if not accel or not gyro:
            return False

        # Store accel in g, convert to m/s^2 for data
        ax, ay, az = accel
        self.data.accel_x = ax * 9.81
        self.data.accel_y = ay * 9.81
        self.data.accel_z = az * 9.81

        # Store gyro with offset correction
        self.data.gyro_x = gyro[0] - self._gyro_offset_x
        self.data.gyro_y = gyro[1] - self._gyro_offset_y
        self.data.gyro_z = gyro[2] - self._gyro_offset_z

        # Compute roll and pitch from accelerometer
        self.data.roll = math.degrees(math.atan2(ay, az))
        self.data.pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))

        self.data.temperature = self._read_temperature()
        self.data.timestamp = time.time()
        self.data.valid = True
        return True

    async def read_loop(self):
        """Async read loop at configured sample rate. Run as asyncio task."""
        if not self._bus:
            if not self.initialize():
                log.error("IMU read_loop aborted: init failed")
                return

        if not self._is_calibrated:
            self.calibrate_gyro()
        self._running = True
        interval = 1.0 / self._sample_rate

        log.info("IMU read_loop started at %dHz", self._sample_rate)
        while self._running:
            t0 = time.monotonic()
            try:
                await asyncio.get_running_loop().run_in_executor(None, self.read)
            except Exception as e:
                log.error("IMU read error: %s", e)

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def stop(self):
        """Signal the read loop to stop."""
        self._running = False
        log.info("IMU reader stopping")

    def close(self):
        """Close the I2C bus."""
        if self._bus:
            self._bus.close()
            self._bus = None
            log.info("IMU I2C bus closed")
