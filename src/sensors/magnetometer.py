"""
GY-273 (HMC5883L) Magnetometer reader via I2C.
Provides magnetic compass heading with declination correction.
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

# HMC5883L Register Map
HMC5883L_REG_CFG_A = 0x00    # Configuration Register A
HMC5883L_REG_CFG_B = 0x01    # Configuration Register B (gain)
HMC5883L_REG_MODE = 0x02     # Mode Register
HMC5883L_REG_DATA_X_MSB = 0x03  # Data output X MSB
HMC5883L_REG_DATA_X_LSB = 0x04
HMC5883L_REG_DATA_Z_MSB = 0x05  # Note: Z comes before Y on HMC5883L!
HMC5883L_REG_DATA_Z_LSB = 0x06
HMC5883L_REG_DATA_Y_MSB = 0x07
HMC5883L_REG_DATA_Y_LSB = 0x08
HMC5883L_REG_STATUS = 0x09
HMC5883L_REG_ID_A = 0x0A     # Identification Register A

# Gain settings (LSB per Gauss)
GAIN_SCALE = {
    1370: 0x00,  # ±0.88 Ga
    1090: 0x20,  # ±1.3 Ga (default)
    820: 0x40,   # ±1.9 Ga
    660: 0x60,   # ±2.5 Ga
    440: 0x80,   # ±4.0 Ga
    390: 0xA0,   # ±4.7 Ga
    330: 0xC0,   # ±5.6 Ga
    230: 0xE0,   # ±8.1 Ga
}


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
    """Reader for GY-273 (HMC5883L) magnetometer via I2C."""

    def __init__(
        self,
        bus: int = HMC5883L_I2C_BUS,
        addr: int = HMC5883L_I2C_ADDR,
        sample_rate_hz: int = HMC5883L_SAMPLE_RATE_HZ,
        declination_deg: float = MAGNETIC_DECLINATION_DEG,
        gain: int = 1090,
    ):
        self._bus_id = bus
        self._addr = addr
        self._sample_rate = sample_rate_hz
        self._declination_deg = declination_deg
        self._gain = gain
        self._bus: Optional[SMBus] = None
        self._gain_scale = 1090.0  # LSB per Gauss
        self.data = MagData()
        self._running = False

        # Hard iron calibration offsets
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._offset_z = 0.0

    def initialize(self) -> bool:
        """Initialize HMC5883L. Returns True on success."""
        try:
            self._bus = SMBus(self._bus_id)

            # Verify identification registers (should read "H43")
            id_a = self._bus.read_byte_data(self._addr, HMC5883L_REG_ID_A)
            if id_a != 0x48:  # 'H'
                log.warning("HMC5883L ID_A=0x%02X (expected 0x48), may be clone", id_a)

            # Configuration Register A:
            # 8 average samples, 15Hz output rate, normal measurement mode
            # Rate bits: 0b100 = 15Hz
            rate_bits = self._get_rate_bits(self._sample_rate)
            cfg_a = 0x60 | (rate_bits << 2)  # 0x60 = 8 samples averaging
            self._bus.write_byte_data(self._addr, HMC5883L_REG_CFG_A, cfg_a)

            # Configuration Register B: Gain = ±1.3 Ga (default, good for most uses)
            gain_reg = GAIN_SCALE.get(self._gain, 0x20)
            self._bus.write_byte_data(self._addr, HMC5883L_REG_CFG_B, gain_reg)
            self._gain_scale = float(self._gain)

            # Mode Register: Continuous measurement mode
            self._bus.write_byte_data(self._addr, HMC5883L_REG_MODE, 0x00)

            time.sleep(0.1)  # Wait for first measurement
            log.info(
                "HMC5883L initialized: rate=%dHz, gain=%d LSB/Ga, decl=%.1f deg",
                self._sample_rate, self._gain, self._declination_deg,
            )
            return True

        except Exception as e:
            log.error("HMC5883L init failed: %s", e)
            return False

    def _get_rate_bits(self, rate_hz: int) -> int:
        """Map sample rate to register bits."""
        rates = {
            0: 0b000, 1: 0b001, 3: 0b010, 7: 0b011,
            15: 0b100, 30: 0b101, 75: 0b110,
        }
        # Find closest supported rate
        closest = min(rates.keys(), key=lambda r: abs(r - rate_hz))
        return rates[closest]

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
            # Read 6 bytes: X_MSB, X_LSB, Z_MSB, Z_LSB, Y_MSB, Y_LSB
            raw = self._bus.read_i2c_block_data(self._addr, HMC5883L_REG_DATA_X_MSB, 6)
            x = struct.unpack(">h", bytes(raw[0:2]))[0] / self._gain_scale
            z = struct.unpack(">h", bytes(raw[2:4]))[0] / self._gain_scale
            y = struct.unpack(">h", bytes(raw[4:6]))[0] / self._gain_scale
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
