"""Sensor contract — dataclass sampel per sensor.

Kontrak antara pembaca sensor (live/replay/synthetic) dan lapisan ML.
Satu pembaca per sensor; data di-fan-out read-only non-blocking ke
konsumen (heading + ML). Window dihitung dengan time.monotonic().
Wall-clock hanya label; Pi tidak punya RTC — tandai waktu belum
tersinkron NTP.
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class SensorSample:
    """Satu bacaan sensor generik. Subclass per sensor untuk field bertipe."""
    timestamp_mono: float        # time.monotonic(), untuk window/buffer
    timestamp_wall: float        # time.time(), label saja (belum NTP sync?)
    ntp_synced: bool = False     # True kalau NTP sudah konvergen
    valid: bool = True
    quality: Optional[float] = None  # GPS fix/HDOP, dll; None jika tidak relevan


@dataclass(slots=True)
class IMUSample(SensorSample):
    """MPU6050 raw. Unit konsisten training: m/s^2, deg/s."""
    accel_x: float = float("nan")
    accel_y: float = float("nan")
    accel_z: float = float("nan")
    gyro_x: float = float("nan")
    gyro_y: float = float("nan")
    gyro_z: float = float("nan")
    roll_deg: float = float("nan")
    pitch_deg: float = float("nan")


@dataclass(slots=True)
class GPSSample(SensorSample):
    """NEO-6M. fix_quality: 0=no fix,1=GPS,2=DGPS."""
    latitude: float = float("nan")
    longitude: float = float("nan")
    speed_ms: float = float("nan")
    course_deg: float = float("nan")
    hdop: float = float("nan")       # horizontal dilution (GGA)
    num_satellites: int = 0
    fix_quality: int = 0


@dataclass(slots=True)
class PressureSample(SensorSample):
    """HX710B. counts raw 24-bit two's complement; kpa terkalibrasi."""
    counts: int = 0
    kpa: float = float("nan")        # setelah offset+scale
    head_m: float = float("nan")     # kpa -> meter kolom air


@dataclass(slots=True)
class WindSample(SensorSample):
    """Anemometer. rps = rotasi/detik; ms terkalibrasi."""
    rps: float = float("nan")
    speed_ms: float = float("nan")


@dataclass(slots=True)
class RainSample(SensorSample):
    """Tipping bucket. tips kumulatif; mm terkalibrasi."""
    tips: int = 0
    rain_mm: float = 0.0


def now_mono() -> float:
    return time.monotonic()


def now_wall() -> float:
    return time.time()
