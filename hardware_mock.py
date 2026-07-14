"""
hardware_mock.py
================
Replaces semua library yang cuma ada di RPi / Linux.
Taruh import ini DI ATAS import hardware library asli.

Cara pakai di main.py / sensor files:

    import sys
    if sys.platform == 'win32':
        from sim.hardware_mock import install
        install()

    # Setelah itu import biasa seperti biasa
    import RPi.GPIO as GPIO   # → pakai mock
    import smbus2             # → pakai mock
    import pigpio             # → pakai mock
"""

import sys
import math
import time
import random
from unittest.mock import MagicMock


# ─────────────────────────────────────────────────────
# RPi.GPIO mock
# ─────────────────────────────────────────────────────
class _GPIO:
    BCM = "BCM"
    BOARD = "BOARD"
    OUT = "OUT"
    IN = "IN"
    HIGH = 1
    LOW = 0
    PUD_UP = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"
    RISING = "RISING"
    FALLING = "FALLING"
    BOTH = "BOTH"

    _pin_state = {}

    def setmode(self, mode): pass
    def setwarnings(self, flag): pass
    def setup(self, pin, mode, pull_up_down=None): self._pin_state[pin] = 0
    def output(self, pin, value): self._pin_state[pin] = value
    def input(self, pin): return self._pin_state.get(pin, 0)
    def cleanup(self): self._pin_state.clear()
    def add_event_detect(self, *a, **kw): pass
    def remove_event_detect(self, pin): pass

    class PWM:
        def __init__(self, pin, freq): self.pin = pin; self.freq = freq
        def start(self, dc): pass
        def ChangeDutyCycle(self, dc): pass
        def ChangeFrequency(self, freq): pass
        def stop(self): pass


# ─────────────────────────────────────────────────────
# pigpio mock (PWM library yang lebih presisi)
# ─────────────────────────────────────────────────────
class _pigpio_pi:
    """Mock pigpio.pi() instance."""
    def __init__(self):
        self._servo_pw = {}   # pin → pulsewidth
        self._pwm = {}
        self.connected = True
        self._bb_pins = set()
        self._bb_accum = 0.0   # accumulator for timed NMEA generation

    def set_servo_pulsewidth(self, pin, pulsewidth):
        self._servo_pw[pin] = pulsewidth

    def get_servo_pulsewidth(self, pin):
        return self._servo_pw.get(pin, 0)

    def set_PWM_dutycycle(self, pin, dc): self._pwm[pin] = dc
    def set_PWM_frequency(self, pin, freq): pass
    def set_mode(self, pin, mode): pass
    def write(self, pin, level): pass
    def read(self, pin): return 0
    def stop(self): pass

    # ── bit-bang serial mock (GPS) ──────────────────────
    def bb_serial_read_open(self, gpio, baud, data_bits=8):
        self._bb_pins.add(gpio)
        return 0

    def bb_serial_read(self, gpio):
        """Return fake NMEA GPGGA once per ~100ms call cadence."""
        self._bb_accum += 1
        if self._bb_accum < 2:          # produce data every other call (~100ms)
            return 0, b""
        self._bb_accum = 0

        lat = -6.8969 + random.gauss(0, 0.000005)
        lon = 112.0521 + random.gauss(0, 0.000005)
        lat_deg = int(abs(lat))
        lat_min = (abs(lat) - lat_deg) * 60
        lon_deg = int(abs(lon))
        lon_min = (abs(lon) - lon_deg) * 60
        ns = 'S' if lat < 0 else 'N'
        ew = 'E' if lon > 0 else 'W'
        sentence = (f"$GPGGA,{time.strftime('%H%M%S.00')},"
                    f"{lat_deg:02d}{lat_min:09.6f},{ns},"
                    f"{lon_deg:03d}{lon_min:09.6f},{ew},"
                    f"1,08,1.0,10.0,M,0,M,,*47\r\n")
        raw = sentence.encode()
        return len(raw), raw

    def bb_serial_read_close(self, gpio):
        self._bb_pins.discard(gpio)
        return 0


class _pigpio:
    INPUT = 0; OUTPUT = 1; ALT0 = 4
    def pi(self): return _pigpio_pi()


# ─────────────────────────────────────────────────────
# smbus2 mock (I2C — untuk IMU, magnetometer)
# ─────────────────────────────────────────────────────
class _SMBus:
    def __init__(self, bus=1): pass
    def write_byte_data(self, addr, reg, val): pass
    def write_i2c_block_data(self, addr, reg, data): pass
    def read_byte_data(self, addr, reg): return 0
    def read_i2c_block_data(self, addr, reg, length): return [0] * length
    def read_word_data(self, addr, reg): return 0
    def close(self): pass

class _smbus2:
    SMBus = _SMBus


# ─────────────────────────────────────────────────────
# serial mock (GPS UART)
# ─────────────────────────────────────────────────────
class _Serial:
    def __init__(self, *a, **kw):
        self.is_open = True
        self._buf = []

    def readline(self):
        # Return fake NMEA GPGGA sentence
        lat = -6.8969 + random.gauss(0, 0.000005)
        lon = 112.0521 + random.gauss(0, 0.000005)
        lat_deg = int(abs(lat))
        lat_min = (abs(lat) - lat_deg) * 60
        lon_deg = int(abs(lon))
        lon_min = (abs(lon) - lon_deg) * 60
        ns = 'S' if lat < 0 else 'N'
        ew = 'E' if lon > 0 else 'W'
        sentence = (f"$GPGGA,{time.strftime('%H%M%S.00')},"
                    f"{lat_deg:02d}{lat_min:09.6f},{ns},"
                    f"{lon_deg:03d}{lon_min:09.6f},{ew},"
                    f"1,08,1.0,10.0,M,0,M,,*47\r\n")
        time.sleep(0.1)
        return sentence.encode()

    def write(self, data): pass
    def flush(self): pass
    def close(self): self.is_open = False
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

class _pyserial:
    Serial = _Serial


# ─────────────────────────────────────────────────────
# Install semua mock ke sys.modules
# ─────────────────────────────────────────────────────
def install():
    """
    Inject semua hardware mock ke sys.modules.
    Panggil ini sebelum import library hardware apapun.
    """
    sys.modules.setdefault('RPi',        MagicMock())
    sys.modules['RPi.GPIO']  = _GPIO()
    sys.modules['pigpio']    = _pigpio()
    sys.modules['smbus2']    = _smbus2()
    sys.modules['serial']    = _pyserial()

    # Library lain yang mungkin dipakai
    for lib in ['board', 'busio', 'adafruit_bno055',
                'adafruit_gps', 'adafruit_lis3mdl']:
        sys.modules.setdefault(lib, MagicMock())

    print("[SIM] Hardware mock installed — running in Windows simulation mode")