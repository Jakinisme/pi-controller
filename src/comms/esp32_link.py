"""
ESP32 Serial Communication Link.

Protocol: JSON frames over serial (USB/UART).
Example frame: {"s": 0.4, "sw": -0.2, "y": 0.1, "ts": 123456.789}

Heartbeat: RPi sends commands at ~20Hz. If ESP32 receives nothing for 1s,
it triggers failsafe (all motors off).
"""

import asyncio
import json
import time
from typing import Optional

import serial

from src.control.thrust import ThrustCommand
from src.config import (
    ESP32_SERIAL_PORT,
    ESP32_BAUD_RATE,
    ESP32_HEARTBEAT_INTERVAL_S,
    ESP32_TIMEOUT_S,
)
from src.utils.logger import setup_logger

log = setup_logger("esp32")

# Reconnect back-off settings
_RECONNECT_INITIAL_S = 2.0
_RECONNECT_MAX_S = 30.0


class ESP32Link:
    """Serial communication link to ESP32 motor controller."""

    def __init__(
        self,
        port: str = ESP32_SERIAL_PORT,
        baud: int = ESP32_BAUD_RATE,
        heartbeat_interval: float = ESP32_HEARTBEAT_INTERVAL_S,
    ):
        self._port = port
        self._baud = baud
        self._heartbeat_interval = heartbeat_interval
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._last_send_time = 0.0
        self._connected = False
        self._bytes_sent = 0
        self._send_errors = 0

        # Latest telemetry from ESP32 (if it sends back status)
        self.esp32_status: dict = {}

    def open(self) -> bool:
        """Open serial port to ESP32."""
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=0.1,
                write_timeout=0.1,
            )
            time.sleep(0.5)  # Wait for ESP32 to reset after serial open
            self._connected = True
            log.info("ESP32 link opened on %s @ %d baud", self._port, self._baud)
            return True
        except serial.SerialException as e:
            log.error("Failed to open ESP32 port %s: %s", self._port, e)
            self._connected = False
            return False

    def close(self):
        """Close serial port."""
        if self._serial and self._serial.is_open:
            # Send zero thrust before closing (safety)
            self.send_thrust(ThrustCommand())
            time.sleep(0.05)
            self._serial.close()
            self._connected = False
            log.info("ESP32 link closed")

    def send_thrust(self, cmd: ThrustCommand) -> bool:
        """Send a thrust command to ESP32.
        
        Args:
            cmd: ThrustCommand with surge, sway, yaw values.
        
        Returns:
            True if sent successfully.
        """
        if not self._serial or not self._serial.is_open:
            self._connected = False
            return False

        payload = {
            "s": round(cmd.surge, 3),
            "sw": round(cmd.sway, 3),
            "y": round(cmd.yaw, 3),
            "ts": round(cmd.timestamp, 3),
        }

        try:
            msg = json.dumps(payload, separators=(",", ":")) + "\n"
            self._serial.write(msg.encode("ascii"))
            self._last_send_time = time.time()
            self._bytes_sent += len(msg)
            return True
        except serial.SerialTimeoutException:
            self._send_errors += 1
            log.warning("ESP32 send timeout (errors: %d)", self._send_errors)
            return False
        except (serial.SerialException, OSError) as e:
            self._send_errors += 1
            self._connected = False
            log.error("ESP32 serial lost: %s (errors: %d)", e, self._send_errors)
            # Close the dead handle so is_connected reflects reality
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
            return False
        except Exception as e:
            self._send_errors += 1
            log.error("ESP32 send error: %s (errors: %d)", e, self._send_errors)
            return False

    def _read_response(self):
        """Read any response from ESP32 (non-blocking)."""
        if not self._serial or not self._serial.in_waiting:
            return
        try:
            line = self._serial.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("{"):
                self.esp32_status = json.loads(line)
        except Exception:
            pass

    async def heartbeat_loop(self):
        """Send periodic heartbeat even when no thrust commands are active.
        
        This keeps the ESP32 from triggering failsafe.
        Sends zero-thrust frames at the heartbeat interval.
        Automatically reconnects if the serial link is lost.
        """
        self._running = True
        reconnect_delay = _RECONNECT_INITIAL_S
        log.info("ESP32 heartbeat loop started (interval=%.2fs)", self._heartbeat_interval)

        while self._running:
            # ---- Reconnect gate ----
            if not self.is_connected:
                log.warning(
                    "ESP32 disconnected — attempting reconnect in %.1fs",
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                if self.open():
                    reconnect_delay = _RECONNECT_INITIAL_S  # reset back-off
                    log.info("ESP32 reconnected successfully")
                else:
                    # Exponential back-off, capped
                    reconnect_delay = min(reconnect_delay * 2, _RECONNECT_MAX_S)
                continue

            # ---- Normal heartbeat ----
            now = time.time()
            if (now - self._last_send_time) >= self._heartbeat_interval:
                cmd = ThrustCommand(surge=0.0, sway=0.0, yaw=0.0, timestamp=now)
                self.send_thrust(cmd)

            # Read any responses from ESP32
            await asyncio.get_running_loop().run_in_executor(None, self._read_response)

            await asyncio.sleep(self._heartbeat_interval / 2)

    def stop(self):
        self._running = False
        log.info("ESP32 link stopping")

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    @property
    def last_send_age(self) -> float:
        """Seconds since last command was sent."""
        if self._last_send_time == 0:
            return float("inf")
        return time.time() - self._last_send_time
