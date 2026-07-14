"""
NEO-6M GPS Reader via pigpio bit-bang UART on GPIO.
Parses NMEA sentences for position, speed, and heading.

Instead of pyserial, we use pigpio's bb_serial_read_open to
receive GPS data on any GPIO pin (default: GPIO23).
Requires pigpiod running on the RPi.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import pigpio
import pynmea2

from src.config import GPS_BB_GPIO, GPS_BAUD_RATE, GPS_TIMEOUT_S
from src.utils.logger import setup_logger

log = setup_logger("gps")

# Maximum buffer size before forced clear (prevents memory leak on noise/bad wiring)
_MAX_BUFFER_BYTES = 2048


@dataclass
class GPSData:
    """Container for parsed GPS data."""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed_over_ground: float = 0.0   # m/s (converted from knots)
    true_course: float = 0.0         # degrees
    fix_quality: int = 0             # 0=no fix, 1=GPS, 2=DGPS
    num_satellites: int = 0
    timestamp: float = 0.0           # Unix timestamp of last valid fix
    valid: bool = False


class GPSReader:
    """Async reader for NEO-6M GPS module via pigpio bit-bang UART."""

    def __init__(
        self,
        gpio: int = GPS_BB_GPIO,
        baud: int = GPS_BAUD_RATE,
        timeout: float = GPS_TIMEOUT_S,
    ):
        self._gpio = gpio
        self._baud = baud
        self._timeout = timeout
        self._pi: Optional[pigpio.pi] = None
        self._bb_open = False
        self.data = GPSData()
        self._running = False
        self._buffer = b""

    def open(self) -> bool:
        """Connect to pigpiod and open bit-bang serial on GPIO. Returns True on success."""
        try:
            self._pi = pigpio.pi()
            if not self._pi.connected:
                log.error("Cannot connect to pigpiod — is the daemon running?")
                return False

            # Open bit-bang serial: gpio, baud, data_bits (8)
            self._pi.bb_serial_read_open(self._gpio, self._baud, 8)
            self._bb_open = True
            log.info(
                "GPS bit-bang opened on GPIO%d @ %d baud (pigpio)",
                self._gpio, self._baud,
            )
            return True
        except Exception as e:
            log.error("Failed to open GPS bit-bang on GPIO%d: %s", self._gpio, e)
            return False

    def close(self):
        """Close the bit-bang serial and disconnect from pigpiod."""
        if self._pi:
            if self._bb_open:
                try:
                    self._pi.bb_serial_read_close(self._gpio)
                except Exception:
                    pass
                self._bb_open = False
            self._pi.stop()
            self._pi = None
            log.info("GPS pigpio connection closed")

    async def read_loop(self):
        """Continuously read and parse NMEA sentences. Run as asyncio task."""
        self._running = True
        if not self._bb_open:
            if not self.open():
                log.error("GPS read_loop aborted: bit-bang not open")
                return

        log.info("GPS read_loop started")
        while self._running:
            try:
                # Poll bit-bang buffer every 50ms
                count, data = self._pi.bb_serial_read(self._gpio)
                if count > 0:
                    self._buffer += data[:count]

                    # Safety valve: clear buffer if too large (noise / broken wire)
                    if len(self._buffer) > _MAX_BUFFER_BYTES:
                        log.warning(
                            "GPS buffer overflow (%d bytes), clearing", len(self._buffer)
                        )
                        self._buffer = b""
                        await asyncio.sleep(0.05)
                        continue

                    # Split into complete lines
                    while b"\n" in self._buffer:
                        line, self._buffer = self._buffer.split(b"\n", 1)
                        sentence = line.decode("ascii", errors="ignore").strip()
                        if sentence:
                            self._parse_nmea(sentence)

            except Exception as e:
                log.error("GPS read error: %s", e)

            await asyncio.sleep(0.05)

        self.close()

    def _parse_nmea(self, line: str):
        """Parse a single NMEA sentence and update GPSData."""
        if not line.startswith("$"):
            return

        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            return

        now = time.time()

        if isinstance(msg, pynmea2.types.talker.GGA):
            self.data.fix_quality = int(msg.gps_qual or 0)
            self.data.num_satellites = int(msg.num_sats or 0)
            self.data.altitude = float(msg.altitude) if msg.altitude else 0.0

            if self.data.fix_quality > 0 and msg.latitude != 0.0:
                self.data.latitude = msg.latitude
                self.data.longitude = msg.longitude
                self.data.timestamp = now
                self.data.valid = True

        elif isinstance(msg, pynmea2.types.talker.RMC):
            if msg.status == "A":  # Active/valid
                self.data.latitude = msg.latitude
                self.data.longitude = msg.longitude
                # Convert knots to m/s
                self.data.speed_over_ground = float(msg.spd_over_grnd or 0) * 0.514444
                self.data.true_course = float(msg.true_course or 0)
                self.data.timestamp = now
                self.data.valid = True

    def stop(self):
        """Signal the read loop to stop."""
        self._running = False
        log.info("GPS reader stopping")

    @property
    def is_connected(self) -> bool:
        return self._pi is not None and self._bb_open

    @property
    def has_fix(self) -> bool:
        return self.data.valid and self.data.fix_quality > 0

    @property
    def data_age(self) -> float:
        """Seconds since last valid GPS fix."""
        if self.data.timestamp == 0:
            return float("inf")
        return time.time() - self.data.timestamp

    @property
    def is_stale(self) -> bool:
        return self.data_age > self._timeout
