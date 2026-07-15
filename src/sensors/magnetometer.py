"""
QMC5883P Magnetometer reader via I2C.
Provides magnetic compass heading with declination correction.

Note: Many GY-273 boards labeled "HMC5883L" actually contain a QMC5883P
chip (I2C addr 0x2C) with a completely different register map.
"""

import asyncio
import time
import math
import struct
from dataclasses import dataclass
from typing import Optional

from smbus2 import SMBus

from src.config import HMC5883L_I2C_BUS, HMC5883L_I2C_ADDR, HMC5883L_SAMPLE_RATE_HZ, MAGNETIC_DECLINATION_DEG
from src.utils.logger import setup_logger

log = setup_logger("mag")

# QMC5883P Register Map
QMC5883P_REG_CHIP_ID = 0x00      # Chip ID (default: 0x80)
QMC5883P_REG_DATA_X_LSB = 0x01   # X-axis output data LSB
QMC5883P_REG_DATA_X_MSB = 0x02   # X-axis output data MSB
QMC5883P_REG_DATA_Y_LSB = 0x03   # Y-axis output data LSB
QMC5883P_REG_DATA_Y_MSB = 0x04   # Y-axis output data MSB
QMC5883P_REG_DATA_Z_LSB = 0x05   # Z-axis output data LSB
QMC5883P_REG_DATA_Z_MSB = 0x06   # Z-axis output data MSB
QMC5883P_REG_STATUS = 0x09       # Status register
QMC5883P_REG_CTRL1 = 0x0A        # Control register 1 (OSR, ODR, Mode)
QMC5883P_REG_CTRL2 = 0x0B        # Control register 2 (Reset, Range)

# Expected chip ID
QMC5883P_CHIP_ID = 0x80

# Control Register 1 (0x0A) bit layout:
#   Bits [1:0] - Mode:  00=Standby, 01=Normal, 10=Single, 11=Continuous
#   Bits [3:2] - ODR:   00=10Hz, 01=50Hz, 10=100Hz, 11=200Hz
#   Bits [5:4] - RNG:   00=2G, 01=8G
#   Bits [7:6] - OSR:   00=512, 01=256, 10=128, 11=64
ODR_MAP = {
    10:  0b00,
    50:  0b01,
    100: 0b10,
    200: 0b11,
}

# Sensitivity (LSB per Gauss) for each range
# Range ±2G: 12000 LSB/G, Range ±8G: 3000 LSB/G
RANGE_2G_SCALE = 12000.0
RANGE_8G_SCALE = 3000.0


@dataclass
class MagData:
    """Container for magnetometer data."""
    x: float = 0.0        # Raw magnetic field X (Gauss)
    y: float = 0.0        # Raw magnetic field Y (Gauss)
    z: float = 0.0        # Raw magnetic field Z (Gauss)
    heading: float = 0.0  # Magnetic heading in degrees [0, 360)
    timestamp: float = 0.0
    valid: bool = False


class HMC5883LReader:
    """Reader for GY-273 magnetometer via I2C.

    Despite the class name (kept for backward-compat with main.py imports),
    this driver targets the QMC5883P chip found on most modern GY-273 boards.
    """

    def __init__(
        self,
        bus: int = HMC5883L_I2C_BUS,
        addr: int = HMC5883L_I2C_ADDR,
        sample_rate_hz: int = HMC5883L_SAMPLE_RATE_HZ,
        declination_deg: float = MAGNETIC_DECLINATION_DEG,
    ):
        self._bus_id = bus
        self._addr = addr
        self._sample_rate = sample_rate_hz
        self._declination_deg = declination_deg
        self._bus: Optional[SMBus] = None
        self._scale = RANGE_8G_SCALE  # LSB per Gauss (default 8G range)
        self.data = MagData()
        self._running = False

        # Hard iron calibration offsets
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._offset_z = 0.0

    def initialize(self) -> bool:
        """Initialize QMC5883P. Returns True on success."""
        try:
            self._bus = SMBus(self._bus_id)

            # Verify chip ID
            chip_id = self._bus.read_byte_data(self._addr, QMC5883P_REG_CHIP_ID)
            if chip_id != QMC5883P_CHIP_ID:
                log.warning(
                    "QMC5883P CHIP_ID=0x%02X (expected 0x%02X), may be different variant",
                    chip_id, QMC5883P_CHIP_ID,
                )

            # Control Register 2 (0x0B): Soft reset, then configure
            # Bit 7 = SOFT_RST, Bit 3 = SET/RESET enable (required)
            self._bus.write_byte_data(self._addr, QMC5883P_REG_CTRL2, 0x80)  # soft reset
            time.sleep(0.05)

            # Control Register 2: Range = 8G (bit 4 = 1), Set/Reset enabled (bit 3 = 1)
            # 0x18 = 0001_1000 → RNG=8G, SET_RESET=on
            self._bus.write_byte_data(self._addr, QMC5883P_REG_CTRL2, 0x08)
            self._scale = RANGE_8G_SCALE

            # Control Register 1 (0x0A): OSR=512, ODR, Mode=Continuous
            # Pick closest supported ODR
            odr_bits = self._get_odr_bits(self._sample_rate)
            # OSR=512 (bits [7:6] = 00), ODR (bits [3:2]), Mode=Continuous (bits [1:0] = 11)
            ctrl1 = (0b00 << 6) | (odr_bits << 2) | 0b11
            self._bus.write_byte_data(self._addr, QMC5883P_REG_CTRL1, ctrl1)

            time.sleep(0.1)  # Wait for first measurement
            log.info(
                "QMC5883P initialized: rate=%dHz, range=8G, decl=%.1f° (addr=0x%02X)",
                self._sample_rate, self._declination_deg, self._addr,
            )
            return True

        except Exception as e:
            log.error("QMC5883P init failed: %s", e)
            if self._bus:
                try:
                    self._bus.close()
                except Exception:
                    pass
                self._bus = None
            return False

    def _get_odr_bits(self, rate_hz: int) -> int:
        """Map sample rate to QMC5883P ODR register bits."""
        # Find closest supported ODR
        closest = min(ODR_MAP.keys(), key=lambda r: abs(r - rate_hz))
        return ODR_MAP[closest]

    def calibrate_hard_iron(self, duration_s: float = 10.0):
        """Calibrate hard iron offsets by rotating the sensor.

        Rotate the ASV/vehicle in a full circle during calibration.
        Call this when you can physically rotate the vehicle.
        """
        log.info("Hard iron calibration: rotate vehicle for %.0f seconds...", duration_s)
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")
        interval = 1.0 / self._sample_rate
        t0 = time.monotonic()

        while (time.monotonic() - t0) < duration_s:
            raw = self._read_raw()
            if raw:
                x, y, z = raw
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
            time.sleep(interval)

        self._offset_x = (min_x + max_x) / 2
        self._offset_y = (min_y + max_y) / 2
        self._offset_z = (min_z + max_z) / 2

        log.info(
            "Hard iron offsets: x=%.4f, y=%.4f, z=%.4f Gauss",
            self._offset_x, self._offset_y, self._offset_z,
        )

    def _read_raw(self):
        """Read raw magnetometer values. Returns (x, y, z) in Gauss."""
        try:
            # Check data ready
            status = self._bus.read_byte_data(self._addr, QMC5883P_REG_STATUS)
            if not (status & 0x01):  # DRDY bit
                return None

            # Read 6 bytes starting at X_LSB: X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB
            raw = self._bus.read_i2c_block_data(self._addr, QMC5883P_REG_DATA_X_LSB, 6)
            x = struct.unpack("<h", bytes(raw[0:2]))[0] / self._scale
            y = struct.unpack("<h", bytes(raw[2:4]))[0] / self._scale
            z = struct.unpack("<h", bytes(raw[4:6]))[0] / self._scale
            return x, y, z
        except Exception as e:
            log.debug("Mag read error: %s", e)
            return None

    def read(self) -> bool:
        """Perform a single read, updating heading. Returns True on success."""
        raw = self._read_raw()
        if not raw:
            return False

        x, y, z = raw

        # Apply hard iron calibration
        x -= self._offset_x
        y -= self._offset_y
        z -= self._offset_z

        self.data.x = x
        self.data.y = y
        self.data.z = z

        # Calculate heading from X and Y (assuming sensor is level)
        heading_rad = math.atan2(y, x)
        heading_deg = math.degrees(heading_rad)

        # Apply magnetic declination
        heading_deg += self._declination_deg

        # Normalize to [0, 360)
        heading_deg = heading_deg % 360

        self.data.heading = heading_deg
        self.data.timestamp = time.time()
        self.data.valid = True
        return True

    async def read_loop(self):
        """Async read loop at configured sample rate."""
        if not self._bus:
            if not self.initialize():
                log.error("Mag read_loop aborted: init failed")
                return

        self._running = True
        interval = 1.0 / self._sample_rate

        log.info("Magnetometer read_loop started at %dHz", self._sample_rate)
        while self._running:
            t0 = time.monotonic()
            try:
                await asyncio.get_running_loop().run_in_executor(None, self.read)
            except Exception as e:
                log.error("Mag read error: %s", e)

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def stop(self):
        self._running = False
        log.info("Magnetometer reader stopping")

    def close(self):
        if self._bus:
            self._bus.close()
            self._bus = None
            log.info("Magnetometer I2C bus closed")
