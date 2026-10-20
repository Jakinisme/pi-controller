"""Synthetic generator — sinyal palsu untuk test/replay.

Pipeline sama untuk live/replay/synthetic; hanya sumber data beda.
Sinus accel dengan frekuensi diketahui, track GPS statis + noise,
pulsa angin/hujan. Dipakai test tanpa hardware.
"""
import math
import time
from typing import Iterator

import numpy as np

from . import config_ml as cfg
from .sensor_contract import (
    IMUSample, GPSSample, PressureSample, WindSample, RainSample,
    now_mono, now_wall,
)


def synthetic_imu_stream(fs: float = 10.0, freq_hz: float = 0.15,
                         amp_ms2: float = 0.20, duration_s: float = 120.0,
                         ) -> Iterator[IMUSample]:
    """Sinus accel vertikal pada freq_hz + noise. fs Hz selama duration_s."""
    n = int(fs * duration_s)
    rng = np.random.default_rng(0)
    t0 = now_mono()
    w0 = now_wall()
    for i in range(n):
        t = i / fs
        mono = t0 + t
        az = 9.81 + amp_ms2 * math.sin(2 * math.pi * freq_hz * t) + rng.normal(0, 0.01)
        ax = 0.05 * math.sin(2 * math.pi * freq_hz * t + 0.3)
        ay = 0.04 * math.sin(2 * math.pi * freq_hz * t + 1.2)
        gx, gy, gz = rng.normal(0, 1e-3, 3)
        yield IMUSample(
            timestamp_mono=mono, timestamp_wall=w0 + t, ntp_synced=False,
            valid=True,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            roll_deg=math.degrees(math.atan2(ay, az)),
            pitch_deg=math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2))),
        )


def synthetic_gps_stream(lat: float = -6.8969, lon: float = 112.0521,
                         drift_m: float = 0.5, duration_s: float = 120.0,
                         fs: float = 1.0, fix_quality: int = 1,
                         hdop: float = 1.0) -> Iterator[GPSSample]:
    """Track statis + noise beberapa meter. fs Hz."""
    n = int(fs * duration_s)
    rng = np.random.default_rng(1)
    t0 = now_mono(); w0 = now_wall()
    dlat = drift_m / 111195.0
    for i in range(n):
        t = i / fs
        yield GPSSample(
            timestamp_mono=t0 + t, timestamp_wall=w0 + t, ntp_synced=False,
            valid=True, quality=hdop,
            latitude=lat + rng.normal(0, dlat * 0.1),
            longitude=lon + rng.normal(0, dlat * 0.1),
            speed_ms=0.0, course_deg=0.0, hdop=hdop,
            num_satellites=8, fix_quality=fix_quality,
        )


def synthetic_pressure_stream(head_m: float = 0.3, duration_s: float = 120.0,
                              fs: float = 1.0) -> Iterator[PressureSample]:
    """Tekanan konstan -> head_m stabil. fs Hz."""
    n = int(fs * duration_s)
    kpa = (cfg.P_ATM + cfg.RHO_SEAWATER * cfg.G * head_m) / 1000.0
    t0 = now_mono(); w0 = now_wall()
    for i in range(n):
        t = i / fs
        yield PressureSample(
            timestamp_mono=t0 + t, timestamp_wall=w0 + t, ntp_synced=False,
            valid=True, counts=0, kpa=kpa, head_m=head_m,
        )


def synthetic_wind_stream(speed_ms: float = 6.0, duration_s: float = 120.0,
                          fs: float = 0.1) -> Iterator[WindSample]:
    n = int(fs * duration_s)
    t0 = now_mono(); w0 = now_wall()
    for i in range(n):
        t = i / fs
        yield WindSample(
            timestamp_mono=t0 + t, timestamp_wall=w0 + t, ntp_synced=False,
            valid=True, rps=speed_ms, speed_ms=speed_ms,
        )


def synthetic_rain_stream(rain_mmh: float = 0.5, duration_s: float = 120.0,
                          fs: float = 0.1) -> Iterator[RainSample]:
    n = int(fs * duration_s)
    t0 = now_mono(); w0 = now_wall()
    mm_per_sample = rain_mmh * (1.0 / fs) / 3600.0
    for i in range(n):
        t = i / fs
        yield RainSample(
            timestamp_mono=t0 + t, timestamp_wall=w0 + t, ntp_synced=False,
            valid=True, tips=i, rain_mm=mm_per_sample,
        )
