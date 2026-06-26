"""
NEO-6M GPS Reader via UART.
Parses NMEA sentences for position, speed, and heading.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import serial
import pynmea2

from src.config import GPS_SERIAL_PORT, GPS_BAUD_RATE, GPS_TIMEOUT_S
from src.utils.logger import setup_logger

log = setup_logger("gps")


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
    """Async reader for NEO-6M GPS module via UART serial."""

    def __init__(
        self,
        port: str = GPS_SERIAL_PORT,
        baud: int = GPS_BAUD_RATE,
        timeout: float = GPS_TIMEOUT_S,
    ):
        self._port = port
        self._baud = baud
        self._timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self.data = GPSData()
        self._running = False

    def open(self) -> bool:
        """Open the serial port. Returns True on success."""
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            log.info("GPS opened on %s @ %d baud", self._port, self._baud)
            return True
        except serial.SerialException as e:
            log.error("Failed to open GPS port %s: %s", self._port, e)
            return False

    def close(self):
        """Close the serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            log.info("GPS port closed")

    async def read_loop(self):
        """Continuously read and parse NMEA sentences. Run as asyncio task."""
        self._running = True
        if not self._serial or not self._serial.is_open:
            if not self.open():
                log.error("GPS read_loop aborted: port not open")
                return

        log.info("GPS read_loop started")
        while self._running:
            try:
                line = await asyncio.get_running_loop().run_in_executor(
                    None, self._read_line
                )
                if line:
                    self._parse_nmea(line)
            except Exception as e:
                log.error("GPS read error: %s", e)
                await asyncio.sleep(0.1)

        self.close()

    def _read_line(self) -> Optional[str]:
        """Read a single NMEA line from serial (blocking, run in executor)."""
        try:
            if self._serial and self._serial.in_waiting:
                raw = self._serial.readline()
                return raw.decode("ascii", errors="ignore").strip()
        except Exception:
            pass
        time.sleep(0.01)  # Small sleep to avoid busy-wait
        return None

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
        return self._serial is not None and self._serial.is_open

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
