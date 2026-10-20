"""ML integration — fan-out read-only dari sensor reader yang ada.

Satu pembaca per sensor (aturan 4). Data IMU/GPS di-fan-out ke buffer ML
read-only, non-blocking. Tidak ubah perilaku heading/loop kontrol.
ML jalan di task terpisah prioritas rendah (sleep.between cycles).
"""
import asyncio
import time
from typing import Optional

from . import config_ml as cfg
from .buffers import RingBuffer
from .sensor_contract import IMUSample, GPSSample, now_mono, now_wall
from .feature_builder import compute_base
from .ml_runner import MLRunner


class MLFanout:
    """Fan-out data sensor ke buffer ML + aggregate per AGG_INTERVAL."""

    def __init__(self, imu_reader, gps_reader, firebase_bridge=None,
                 model_feed_fn=None):
        self.imu = imu_reader
        self.gps = gps_reader
        self.firebase = firebase_bridge
        self._model_feed_fn = model_feed_fn  # callable() -> dict|None
        self.imu_buf = RingBuffer("imu", cfg.AGG_WINDOW_SECONDS + 60, 50.0)
        self.gps_buf = RingBuffer("gps", cfg.AGG_WINDOW_SECONDS + 60, 1.0)
        # pressure/wind/rain: driver Fase 2 isi buffer ini (stub sekarang)
        self.pressure_buf = RingBuffer("pressure", cfg.AGG_WINDOW_SECONDS + 60, 1.0)
        self.wind_buf = RingBuffer("wind", cfg.AGG_WINDOW_SECONDS + 60, 1.0)
        self.rain_buf = RingBuffer("rain", cfg.AGG_WINDOW_SECONDS + 60, 1.0)
        self._running = False
        self._runner: Optional[MLRunner] = None
        try:
            self._runner = MLRunner()
        except Exception as e:
            # artifacts hilang? tetap jalan tanpa runner, push status degraded
            self._runner = None
            print(f"[ML] runner init failed: {e}")

    async def ml_loop(self):
        """Fan-out IMU/GPS @ ~5Hz + aggregate per AGG_INTERVAL.
        Loop kontrol tidak boleh terganggu — semua beban ML di sini."""
        if not cfg.ML_ENABLED:
            print("[ML] disabled (ML_ENABLED=False)")
            return
        self._running = True
        fanout_hz = 5.0
        fanout_interval = 1.0 / fanout_hz
        last_cycle_mono = now_mono()
        score_state = "ON" if cfg.SCORING_ENABLED else "OFF (shadow)"
        print(f"[ML] ml_loop started, scoring={score_state}, AGG={cfg.AGG_WINDOW_SECONDS}s")
        while self._running:
            t0 = time.monotonic()
            try:
                self._fanout_once()
            except Exception as e:
                print(f"[ML] fanout error: {e}")
            # cek apakah sudah sampai AGG_INTERVAL
            if (time.monotonic() - last_cycle_mono) >= cfg.AGG_WINDOW_SECONDS:
                last_cycle_mono = time.monotonic()
                try:
                    self._run_cycle()
                except Exception as e:
                    print(f"[ML] cycle error: {e}")
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, fanout_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def _fanout_once(self):
        """Baca imu.data/gps.data read-only, append ke buffer ML."""
        mono = now_mono()
        wall = now_wall()
        imu_d = self.imu.data
        if imu_d.valid and not self.imu.is_stale:
            self.imu_buf.append(IMUSample(
                timestamp_mono=mono, timestamp_wall=wall, ntp_synced=False,
                valid=True, accel_x=imu_d.accel_x, accel_y=imu_d.accel_y,
                accel_z=imu_d.accel_z, gyro_x=imu_d.gyro_x, gyro_y=imu_d.gyro_y,
                gyro_z=imu_d.gyro_z, roll_deg=imu_d.roll, pitch_deg=imu_d.pitch,
            ))
        gps_d = self.gps.data
        if gps_d.valid and self.gps.has_fix:
            self.gps_buf.append(GPSSample(
                timestamp_mono=mono, timestamp_wall=wall, ntp_synced=False,
                valid=True, quality=1.0,
                latitude=gps_d.latitude, longitude=gps_d.longitude,
                speed_ms=gps_d.speed_over_ground, course_deg=gps_d.true_course,
                hdop=getattr(gps_d, "hdop", float("nan")),
                num_satellites=gps_d.num_satellites, fix_quality=gps_d.fix_quality,
            ))

    def _run_cycle(self):
        mode = "station_keeping"
        mf = self._model_feed_fn() if self._model_feed_fn else None
        base = compute_base(
            self.imu_buf.snapshot(), self.gps_buf.snapshot(),
            self.pressure_buf.snapshot(), self.wind_buf.snapshot(),
            self.rain_buf.snapshot(),
            model_feed=mf, mode=mode, window_s=cfg.AGG_WINDOW_SECONDS,
        )
        if self._runner is None:
            return
        result = self._runner.cycle(base, mode=mode)
        if self.firebase is not None:
            self.firebase.push_ml(self._runner.telemetry_payload(result))

    def stop(self):
        self._running = False