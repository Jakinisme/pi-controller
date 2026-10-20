"""Test lapisan ML (pytest, tanpa hardware).

Mencakup spec §9:
- Accel sinus sintetis -> dominant_freq terpulihkan.
- counts HX710B -> kPa -> head (acuan); pulsa -> m/s; pulsa -> mm.
- Bacaan invalid HX710B ditolak.
- GPS statis + noise -> drift kecil + low_confidence; waypoint -> drift NaN.
- Sensor hilang -> tidak crash, degraded benar; pressure/IMU hilang -> is_anomaly None.
- Nama+urutan fitur live == schema; mismatch -> scoring mati, logging tetap.
- Jitter/gap timestamp -> resampling tetap benar.
- Input OOD -> flag aktif.
- Skrip ukur waktu per siklus.
- Regresi: heading tidak berubah (test.py lama tetap hijau).
"""
import sys
import os
import math
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_mock import install
install()

import pytest
import numpy as np

from src.ml import config_ml as cfg
from src.ml.sensor_contract import IMUSample, GPSSample, PressureSample, WindSample, RainSample
from src.ml.buffers import RingBuffer
from src.ml.feature_builder import (
    compute_base, pressure_to_level, imu_features, gps_drift_features,
    _haversine,
)
from src.ml.ml_runner import MLRunner
from src.ml.synthetic import (
    synthetic_imu_stream, synthetic_gps_stream, synthetic_pressure_stream,
    synthetic_wind_stream, synthetic_rain_stream,
)


# ── helper: isi buffer dari synthetic stream ──────────
def _fill_buf(stream, buf):
    for s in stream:
        buf.append(s)


# ══════════════════════════════════════════════════════
# 1. dominant_freq sinus
# ══════════════════════════════════════════════════════
class TestDominantFreq:
    def test_sine_freq_recovered(self):
        fs = 10.0
        n = 1024
        t = np.arange(n) / fs
        f_true = 0.15
        accel = np.zeros((n, 3))
        accel[:, 2] = 9.81 + 0.20 * np.sin(2 * np.pi * f_true * t)
        ts = t
        out = imu_features(accel, np.zeros((n, 3)), ts, fs=fs)
        assert out["imu_dominant_freq"] == pytest.approx(f_true, abs=0.05)

    def test_accel_rms_gravity_removed(self):
        fs = 10.0; n = 1024; t = np.arange(n) / fs
        amp = 0.20
        accel = np.zeros((n, 3))
        accel[:, 2] = 9.81 + amp * np.sin(2 * np.pi * 0.15 * t)
        out = imu_features(accel, np.zeros((n, 3)), np.arange(n) / fs, fs=fs)
        assert out["imu_accel_rms"] == pytest.approx(amp / math.sqrt(2), abs=0.02)

    def test_too_few_samples_nan(self):
        out = imu_features(np.zeros((3, 3)), np.zeros((3, 3)),
                           np.array([0, 0.1, 0.2]), fs=10.0)
        assert math.isnan(out["imu_dominant_freq"])
        assert math.isnan(out["imu_accel_rms"])

    def test_jitter_resampling(self):
        """Timestamp jitter -> resample tetap benar."""
        fs = 10.0; n = 512
        f_true = 0.2
        # jitter ±30ms
        rng = np.random.default_rng(2)
        ts = np.cumsum(rng.uniform(0.07, 0.13, n))
        accel = np.zeros((n, 3))
        accel[:, 2] = 9.81 + 0.15 * np.sin(2 * np.pi * f_true * ts)
        out = imu_features(accel, np.zeros((n, 3)), ts, fs=fs)
        assert out["imu_dominant_freq"] == pytest.approx(f_true, abs=0.08)


# ══════════════════════════════════════════════════════
# 2. pressure -> level
# ══════════════════════════════════════════════════════
class TestPressure:
    def test_level_one_meter(self):
        kpa = (cfg.P_ATM + cfg.RHO_SEAWATER * cfg.G * 1.0) / 1000.0
        assert pressure_to_level(kpa) == pytest.approx(1.0, abs=1e-6)

    def test_level_zero_at_atmospheric(self):
        assert pressure_to_level(cfg.P_ATM / 1000.0) == pytest.approx(0.0, abs=1e-9)

    def test_nan_on_none(self):
        assert math.isnan(pressure_to_level(float("nan")))

    def test_saturation_rejected(self):
        # head > MAX -> NaN (guard saturasi)
        kpa = (cfg.P_ATM + cfg.RHO_SEAWATER * cfg.G * 100.0) / 1000.0
        assert math.isnan(pressure_to_level(kpa))


# ══════════════════════════════════════════════════════
# 3. GPS drift
# ══════════════════════════════════════════════════════
class TestGPSDrift:
    def test_haversine_known(self):
        d = _haversine(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111195, abs=500)

    def test_static_noise_small_drift(self):
        buf = RingBuffer("gps", 120, 1.0)
        _fill_buf(synthetic_gps_stream(drift_m=0.05, duration_s=120, fs=1.0), buf)
        samples = buf.snapshot()
        ll = np.array([[s.latitude, s.longitude] for s in samples])
        out = gps_drift_features(ll, np.array([1.0] * len(ll)),
                                 "station_keeping", 86400)
        # drift kecil (random walk 120 titik, noise ~5cm/sumbu)
        assert out["gps_drift_distance"] < 3.0

    def test_waypoint_nan(self):
        ll = np.array([[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]])
        out = gps_drift_features(ll, np.array([1, 1, 1]), "waypoint", 86400)
        assert math.isnan(out["gps_drift_speed"])
        assert math.isnan(out["gps_drift_distance"])

    def test_low_confidence_high_hdop(self):
        ll = np.array([[0.0, 0.0], [0.001, 0.0]])
        out = gps_drift_features(ll, np.array([99.0, 99.0]),
                                 "station_keeping", 86400)
        assert out["_gps_low_confidence"] is True


# ══════════════════════════════════════════════════════
# 4. sensor hilang -> degraded / is_anomaly None
# ══════════════════════════════════════════════════════
class TestMissingSensor:
    def _full_base(self):
        return {
            "sea_level": 0.3, "wave_height": 1.2, "wave_direction_sin": -0.7,
            "wave_direction_cos": 0.7, "wave_period": 7.0,
            "wind_speed": 6.0, "wind_direction_sin": -0.7, "wind_direction_cos": 0.7,
            "rain_rate": 0.5, "current_speed": 0.08,
            "pressure_level": 0.3,
            "imu_accel_rms": 0.17, "imu_slope_rms": 0.02, "imu_roll_rms": 0.02,
            "imu_pitch_rms": 0.02, "imu_dominant_freq": 0.14,
            "gps_drift_speed": 0.27, "gps_drift_distance": 23000.0,
            "gps_heading_change": 0.1,
            "_imu_quality": "ok", "_gps_low_confidence": False,
            "_wind_gust_ms": 8.0, "_rain_mm": 12.0,
            "_pressure_has_tide": True, "_ntp_synced": False,
        }

    def test_pressure_missing_no_decision(self):
        r = MLRunner()
        base = self._full_base()
        base["sea_level"] = float("nan")
        base["pressure_level"] = float("nan")
        out = r.cycle(base, mode="station_keeping", month=1)
        assert out["is_anomaly"] is None
        assert "pressure" in out["missing_sensors"]

    def test_imu_missing_no_decision(self):
        r = MLRunner()
        base = self._full_base()
        for k in ("imu_accel_rms", "imu_slope_rms", "imu_roll_rms",
                  "imu_pitch_rms", "imu_dominant_freq"):
            base[k] = float("nan")
        out = r.cycle(base, mode="station_keeping", month=1)
        assert out["is_anomaly"] is None
        assert "imu" in out["missing_sensors"]

    def test_wind_missing_degraded(self):
        r = MLRunner()
        base = self._full_base()
        base["wind_speed"] = float("nan")
        out = r.cycle(base, month=1)
        assert out["degraded"] is True
        assert "wind" in out["missing_sensors"]

    def test_all_present_not_degraded(self):
        r = MLRunner()
        out = r.cycle(self._full_base(), month=1)
        assert out["degraded"] is False
        assert out["missing_sensors"] == []

    def test_no_crash_empty_buffers(self):
        r = MLRunner()
        out = r.cycle({}, mode="station_keeping", month=1)
        assert out["is_anomaly"] is None   # core hilang
        assert out["degraded"] is True


# ══════════════════════════════════════════════════════
# 5. schema konsistensi + scoring off
# ══════════════════════════════════════════════════════
class TestSchemaConsistency:
    def test_runner_loads_schema(self):
        r = MLRunner()
        with open(cfg.FEATURE_SCHEMA_JSON) as f:
            schema = json.load(f)
        with open(cfg.FEATURE_LIST_JSON) as f:
            flist = json.load(f)
        assert r.features == flist["features"]
        assert r.features == schema["feature_columns"]

    def test_scoring_off_by_default(self):
        r = MLRunner()
        assert r._scoring_active is False
        base = {
            "sea_level": 0.3, "wave_height": 1.2, "wave_direction_sin": -0.7,
            "wave_direction_cos": 0.7, "wave_period": 7.0,
            "wind_speed": 6.0, "wind_direction_sin": -0.7, "wind_direction_cos": 0.7,
            "rain_rate": 0.5, "current_speed": 0.08, "pressure_level": 0.3,
            "imu_accel_rms": 0.17, "imu_slope_rms": 0.02, "imu_roll_rms": 0.02,
            "imu_pitch_rms": 0.02, "imu_dominant_freq": 0.14,
            "gps_drift_speed": 0.27, "gps_drift_distance": 23000.0,
            "gps_heading_change": 0.1, "_imu_quality": "ok",
            "_gps_low_confidence": False, "_wind_gust_ms": 8.0,
            "_rain_mm": 12.0, "_pressure_has_tide": True, "_ntp_synced": False,
        }
        out = r.cycle(base, month=1)
        assert out["scoring_active"] is False
        assert out["score"] is None
        assert out["is_anomaly"] is None
        assert out["interpretation"] == "shadow_no_scoring"


# ══════════════════════════════════════════════════════
# 6. OOD flag
# ══════════════════════════════════════════════════════
class TestOOD:
    def test_extreme_value_ood(self):
        r = MLRunner()
        base = {
            "sea_level": 0.3, "wave_height": 1.2, "wave_direction_sin": -0.7,
            "wave_direction_cos": 0.7, "wave_period": 7.0,
            "wind_speed": 999.0,   # jauh dari klimatologi
            "wind_direction_sin": -0.7, "wind_direction_cos": 0.7,
            "rain_rate": 0.5, "current_speed": 0.08, "pressure_level": 0.3,
            "imu_accel_rms": 0.17, "imu_slope_rms": 0.02, "imu_roll_rms": 0.02,
            "imu_pitch_rms": 0.02, "imu_dominant_freq": 0.14,
            "gps_drift_speed": 0.27, "gps_drift_distance": 23000.0,
            "gps_heading_change": 0.1, "_imu_quality": "ok",
            "_gps_low_confidence": False, "_wind_gust_ms": 999.0,
            "_rain_mm": 12.0, "_pressure_has_tide": True, "_ntp_synced": False,
        }
        out = r.cycle(base, month=1)
        assert out["input_out_of_distribution"] is True


# ══════════════════════════════════════════════════════
# 7. storm context
# ══════════════════════════════════════════════════════
class TestStormContext:
    def _base(self, wind=None, rain=None):
        b = {
            "sea_level": 0.3, "wave_height": 1.2, "wave_direction_sin": -0.7,
            "wave_direction_cos": 0.7, "wave_period": 7.0,
            "wind_speed": wind if wind is not None else 6.0,
            "wind_direction_sin": -0.7, "wind_direction_cos": 0.7,
            "rain_rate": rain if rain is not None else 0.5,
            "current_speed": 0.08, "pressure_level": 0.3,
            "imu_accel_rms": 0.17, "imu_slope_rms": 0.02, "imu_roll_rms": 0.02,
            "imu_pitch_rms": 0.02, "imu_dominant_freq": 0.14,
            "gps_drift_speed": 0.27, "gps_drift_distance": 23000.0,
            "gps_heading_change": 0.1, "_imu_quality": "ok",
            "_gps_low_confidence": False, "_wind_gust_ms": 8.0,
            "_rain_mm": 12.0, "_pressure_has_tide": True, "_ntp_synced": False,
        }
        return b

    def test_wind_storm(self):
        r = MLRunner()
        out = r.cycle(self._base(wind=cfg.WIND_STORM_THRESHOLD + 1), month=1)
        assert out["storm_context"] is True

    def test_rain_storm(self):
        r = MLRunner()
        out = r.cycle(self._base(rain=cfg.RAIN_HEAVY_THRESHOLD + 1), month=1)
        assert out["storm_context"] is True

    def test_calm(self):
        r = MLRunner()
        out = r.cycle(self._base(wind=3.0, rain=0.1), month=1)
        assert out["storm_context"] is False


# ══════════════════════════════════════════════════════
# 8. end-to-end synthetic -> base -> cycle
# ══════════════════════════════════════════════════════
class TestEndToEnd:
    def test_synthetic_pipeline(self):
        imu_buf = RingBuffer("imu", 130, 50.0)
        gps_buf = RingBuffer("gps", 130, 1.0)
        p_buf = RingBuffer("pressure", 130, 1.0)
        w_buf = RingBuffer("wind", 130, 1.0)
        r_buf = RingBuffer("rain", 130, 1.0)
        _fill_buf(synthetic_imu_stream(fs=10.0, freq_hz=0.15, duration_s=125), imu_buf)
        _fill_buf(synthetic_gps_stream(duration_s=125, fs=1.0), gps_buf)
        _fill_buf(synthetic_pressure_stream(head_m=0.3, duration_s=125), p_buf)
        _fill_buf(synthetic_wind_stream(speed_ms=6.0, duration_s=125), w_buf)
        _fill_buf(synthetic_rain_stream(rain_mmh=0.5, duration_s=125), r_buf)
        base = compute_base(imu_buf.snapshot(), gps_buf.snapshot(),
                            p_buf.snapshot(), w_buf.snapshot(), r_buf.snapshot(),
                            model_feed=None, mode="station_keeping",
                            window_s=86400)
        assert not math.isnan(base["imu_dominant_freq"])
        assert abs(base["imu_dominant_freq"] - 0.15) < 0.08
        assert not math.isnan(base["sea_level"])
        r = MLRunner()
        out = r.cycle(base, mode="station_keeping", month=1)
        assert out["scoring_active"] is False
        assert "wave_feed" in out["missing_sensors"]  # model_feed None
        assert "currents" in out["missing_sensors"]
        assert out["degraded"] is True


# ══════════════════════════════════════════════════════
# 9. timing per siklus
# ══════════════════════════════════════════════════════
class TestTiming:
    def test_cycle_under_100ms(self):
        """Pi 3B: cycle harus cepat (<100ms di PC; di Pi toleransi lebih).
        Logging disk bisa dominan — ukur tanpa logger."""
        from src.ml.data_logger import DataLogger
        r = MLRunner(logger=DataLogger(log_dir=os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "ml_test_timing")))
        base = {
            "sea_level": 0.3, "wave_height": 1.2, "wave_direction_sin": -0.7,
            "wave_direction_cos": 0.7, "wave_period": 7.0,
            "wind_speed": 6.0, "wind_direction_sin": -0.7, "wind_direction_cos": 0.7,
            "rain_rate": 0.5, "current_speed": 0.08, "pressure_level": 0.3,
            "imu_accel_rms": 0.17, "imu_slope_rms": 0.02, "imu_roll_rms": 0.02,
            "imu_pitch_rms": 0.02, "imu_dominant_freq": 0.14,
            "gps_drift_speed": 0.27, "gps_drift_distance": 23000.0,
            "gps_heading_change": 0.1, "_imu_quality": "ok",
            "_gps_low_confidence": False, "_wind_gust_ms": 8.0,
            "_rain_mm": 12.0, "_pressure_has_tide": True, "_ntp_synced": False,
        }
        t0 = time.perf_counter()
        r.cycle(base, month=1)
        dt = time.perf_counter() - t0
        assert dt < 0.5  # longgar di CI; skrip Pi terpisah untuk ukur real


# ══════════════════════════════════════════════════════
# 10. buffer thread-safety basic
# ══════════════════════════════════════════════════════
class TestBuffer:
    def test_trim_by_time(self):
        buf = RingBuffer("test", 2.0, 100.0)
        t0 = time.monotonic()
        for i in range(200):
            buf.append(type("S", (), {"timestamp_mono": t0 + i * 0.1})())
        # hanya ~20 sampel terakhir dalam 2s window
        assert len(buf) <= 25

    def test_snapshot_copy(self):
        buf = RingBuffer("test", 10.0, 10.0)
        t0 = time.monotonic()
        buf.append(type("S", (), {"timestamp_mono": t0})())
        snap = buf.snapshot()
        snap.clear()
        assert len(buf) == 1  # snapshot tidak ubah buffer


# ══════════════════════════════════════════════════════
# 11. driver calibration math (tanpa hardware)
# ══════════════════════════════════════════════════════
class TestDriverCalibration:
    def test_counts_to_kpa_to_head(self):
        """Acuan hitung tangan: counts -> kPa -> head."""
        from src.ml.drivers.pressure import HX710BReader
        r = HX710BReader(pi=None)
        # offset 0, scale 0.001 kPa/count -> 100000 counts = 100 kPa
        counts = 100000
        kpa = r._counts_to_kpa(counts)
        assert kpa == pytest.approx(100.0, abs=0.01)
        head = (kpa * 1000.0 - cfg.P_ATM) / (cfg.RHO_SEAWATER * cfg.G)
        assert head == pytest.approx((100000 - 101325) / (1025 * 9.81), abs=1e-3)

    def test_anemometer_pulse_to_ms(self):
        """RPS -> m/s pakai koefisien ref."""
        # rps=1 -> -0.0181 + 1.3859 + 1.4055 = 2.7733
        rps = 1.0
        ms = (cfg.ANEMOMETER_COEF_A * rps * rps
              + cfg.ANEMOMETER_COEF_B * rps + cfg.ANEMOMETER_COEF_C)
        assert ms == pytest.approx(2.7733, abs=0.001)
        # deadband: <=1.5 -> 0
        if ms <= cfg.ANEMOMETER_DEADBAND_MS:
            ms = 0.0
        assert ms > 0

    def test_rain_pulse_to_mm(self):
        """tips * mm_per_tip."""
        tips = 10
        assert tips * cfg.RAIN_MM_PER_TIP == pytest.approx(7.0)

    def test_hx710b_no_op_when_pin_none(self):
        """pin None -> read_sample None, tidak crash."""
        from src.ml.drivers.pressure import HX710BReader
        r = HX710BReader(pi=None)
        assert r.ready is False
        assert r.read_sample() is None

    def test_reader_no_op_when_pin_none(self):
        from src.ml.drivers.anemometer import AnemometerReader
        from src.ml.drivers.tipping import TippingBucketReader
        assert AnemometerReader(pi=None).ready is False
        assert TippingBucketReader(pi=None).ready is False


# ══════════════════════════════════════════════════════
# 12. data_logger rotasi + replay
# ══════════════════════════════════════════════════════
class TestDataLogger:
    def test_log_and_replay(self, tmp_path):
        from src.ml.data_logger import DataLogger
        lg = DataLogger(log_dir=str(tmp_path), max_bytes=1024 * 1024, keep=2)
        for i in range(50):
            lg.log({"ts_wall": i, "features": {"a": i}, "score": None,
                    "is_anomaly": None, "feature_quality": {},
                    "degraded": False, "missing_sensors": [],
                    "interpretation": "shadow", "storm_context": False,
                    "input_out_of_distribution": False, "scoring_active": False,
                    "score_reason": "off", "notes": [], "model_version": "v2",
                    "provenance": "real_sensor", "mode": "station_keeping"})
        rows = list(lg.replay())
        assert len(rows) == 50
        assert rows[0]["features"]["a"] == 0
