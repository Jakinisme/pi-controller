"""Feature builder — dari ring buffer sensor ke dict fitur BASE.

Port dari machine-learning/src/v2/live_features.py dengan perbaikan:
- imu_slope_rms: di training dari add_physics_proxy (k·sigma_eta dari wave);
  di sini dihitung dari model_feed wave (konsisten training physics_proxy).
- resample IMU ke grid waktu seragam (np.interp) karena loop Python punya jitter.
- cek kualitas IMU: min sampel, rate efektif, gap terbesar -> NaN+flag.
- GPS: drift hanya saat station_keeping; gate fix+HDOP; ambang minimum+flag.
- schema training (feature_schema.json) menang; tidak duplikasi daftar fitur.

Nama/satuan/window PERSIS training. Derived (deseasonalized/ma/std/delta/
rate) BUKAN di sini — dihitung ml_runner (rolling history) supaya konsisten.
"""
import math
import json
from typing import Optional

import numpy as np
from scipy.signal import welch

from . import config_ml as cfg
from .sensor_contract import (
    IMUSample, GPSSample, PressureSample, WindSample, RainSample,
)


def pressure_to_level(kpa: float) -> float:
    """kPa -> meter kolom air. (P-P_atm)/(rho*g). Sensor HX710B output kpa."""
    if kpa is None or (isinstance(kpa, float) and math.isnan(kpa)):
        return float("nan")
    pa = kpa * 1000.0
    head = (pa - cfg.P_ATM) / (cfg.RHO_SEAWATER * cfg.G)
    # offset kedalaman ujung selang + guard saturasi
    head -= cfg.TUBE_DEPTH_OFFSET_M
    if abs(head) > cfg.HX710B_MAX_HEAD_M:
        return float("nan")   # saturasi -> invalid
    return head


def _isnan(v) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


# ── IMU ───────────────────────────────────────────────
def imu_features(accel_xyz: np.ndarray, gyro_xyz: np.ndarray, ts: np.ndarray,
                 fs: float = 10.0) -> dict:
    """accel_xyz, gyro_xyz: (N,3). ts: (N,) monotonic seconds (resample grid).
    Resample ke grid seragam (np.interp) karena loop Python punya jitter.
    Return fitur IMU konsisten training (m/s^2, rad, rad, Hz).
    """
    nan = float("nan")
    a = np.asarray(accel_xyz, dtype=float)
    g = np.asarray(gyro_xyz, dtype=float)
    t = np.asarray(ts, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 4 or t.shape[0] != a.shape[0]:
        return dict(imu_accel_rms=nan, imu_slope_rms=nan,
                    imu_roll_rms=nan, imu_pitch_rms=nan, imu_dominant_freq=nan,
                    _imu_quality="too_few_samples")

    # resample ke grid seragam (median dt)
    dt_med = float(np.median(np.diff(t)))
    if dt_med <= 0 or not np.isfinite(dt_med):
        dt_med = 1.0 / fs
    n = len(t)
    grid = t[0] + np.arange(n) * dt_med
    ar = np.empty_like(a)
    for ax in range(3):
        ar[:, ax] = np.interp(grid, t, a[:, ax])
    az = ar[:, 2]
    fs_eff = 1.0 / dt_med

    # gravitasi dibuang: residual vertikal = az - mean(az)
    accel_rms = float(np.sqrt(np.mean((az - az.mean()) ** 2)))
    # roll/pitch dari accel (small-angle), RMS over window (training pakai rad)
    roll = np.arctan2(ar[:, 1], np.sqrt(ar[:, 0] ** 2 + ar[:, 2] ** 2))
    pitch = np.arctan2(-ar[:, 0], np.sqrt(ar[:, 1] ** 2 + ar[:, 2] ** 2))
    roll_rms = float(np.sqrt(np.mean(roll ** 2)))
    pitch_rms = float(np.sqrt(np.mean(pitch ** 2)))
    # dominant freq via Welch pada residual vertikal
    try:
        f, pxx = welch(az - az.mean(), fs=fs_eff, nperseg=min(len(az), 256))
        if len(f) > 1:
            dom = float(f[np.argmax(pxx[1:]) + 1])
        else:
            dom = nan
    except Exception:
        dom = nan
    # imu_slope_rms: di training = k*sigma_eta dari wave (physics_proxy).
    # Tidak bisa dari IMU murni -> diisi di compute_base dari model_feed.
    return dict(imu_accel_rms=accel_rms, imu_slope_rms=nan,
                imu_roll_rms=roll_rms, imu_pitch_rms=pitch_rms,
                imu_dominant_freq=dom, _imu_quality="ok",
                _imu_fs_eff=fs_eff)


# ── GPS ───────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def gps_drift_features(latlon: np.ndarray, hdop: np.ndarray, mode: str,
                       window_s: float) -> dict:
    """latlon: (N,2) track dalam satu AGG_INTERVAL window (yang lock+valid).
    Saat mode != 'station_keeping' -> drift = NaN (per spec).
    Gate fix + HDOP; ambang minimum + flag low_confidence.
    """
    nan = float("nan")
    base = dict(gps_drift_speed=nan, gps_drift_distance=nan,
                gps_heading_change=nan, _gps_low_confidence=False)
    if mode != "station_keeping":
        return base
    ll = np.asarray(latlon, dtype=float)
    if ll.ndim != 2 or ll.shape[0] < 2:
        return base
    dists, headings = [], []
    for i in range(1, len(ll)):
        d = _haversine(ll[i - 1, 0], ll[i - 1, 1], ll[i, 0], ll[i, 1])
        dists.append(d)
        dy, dx = ll[i, 0] - ll[i - 1, 0], ll[i, 1] - ll[i - 1, 1]
        headings.append(math.degrees(math.atan2(dy, dx)))
    total = float(sum(dists))
    # ambang minimum noise GPS
    if total < cfg.GPS_DRIFT_MIN_M:
        total = 0.0
    speed = float(total / max(1.0, window_s))
    hc = 0.0
    for i in range(1, len(headings)):
        diff = abs(headings[i] - headings[i - 1])
        diff = min(diff, 360 - diff)
        hc += diff
    low_conf = False
    if hdop.size and float(np.nanmax(hdop)) > cfg.GPS_MIN_HDOP:
        low_conf = True
    return dict(gps_drift_speed=speed, gps_drift_distance=total,
                gps_heading_change=math.radians(hc),
                _gps_low_confidence=low_conf)


# ── wind/rain aggregate ───────────────────────────────
def wind_features(samples: list) -> dict:
    """mean speed, gust (max) per AGG_INTERVAL. dir: sin/cos sirkular mean."""
    nan = float("nan")
    speeds = [s.speed_ms for s in samples if s.valid and not _isnan(s.speed_ms)]
    if not speeds:
        return dict(wind_speed=nan, wind_direction_sin=nan, wind_direction_cos=nan,
                    wind_gust_ms=nan)
    # anemometer ref tidak ukur arah -> dir NaN (imputed climatology)
    return dict(wind_speed=float(np.mean(speeds)),
                wind_gust_ms=float(np.max(speeds)),
                wind_direction_sin=nan, wind_direction_cos=nan)


def rain_features(samples: list, window_s: float) -> dict:
    """rain_mm dalam window, rain_rate_mmh."""
    nan = float("nan")
    tips = [s for s in samples if s.valid]
    if not tips:
        return dict(rain_mm=nan, rain_rate_mmh=nan)
    total_mm = sum(s.rain_mm for s in tips)
    rate = total_mm / max(1.0, window_s) * 3600.0
    return dict(rain_mm=float(total_mm), rain_rate_mmh=float(rate))


# ── main aggregator ───────────────────────────────────
def compute_base(imu_buf: list, gps_buf: list, pressure_buf: list,
                 wind_buf: list, rain_buf: list,
                 model_feed: Optional[dict], mode: str,
                 window_s: float) -> dict:
    """Hitung dict fitur BASE untuk satu AGG_INTERVAL window.
    Derived (deseasonalized/ma/std/delta/rate) dihitung ml_runner.
    model_feed: {wave_height, wave_period, wave_direction, current_speed} NRT
      atau None -> fitur gelombang/arus NaN (diimpute, degraded).
    mode: 'station_keeping' | 'waypoint'. drift NaN jika bukan station_keeping.
    """
    mf = model_feed or {}
    nan = float("nan")

    # ── IMU ──
    imu_ok = [s for s in imu_buf if s.valid]
    if len(imu_ok) >= cfg.IMU_MIN_SAMPLES:
        ts = np.array([s.timestamp_mono for s in imu_ok])
        accel = np.array([[s.accel_x, s.accel_y, s.accel_z] for s in imu_ok])
        gyro = np.array([[s.gyro_x, s.gyro_y, s.gyro_z] for s in imu_ok])
        imu = imu_features(accel, gyro, ts)
        # quality gate: rate & gap
        if len(ts) >= 2:
            dt = np.diff(ts)
            rate_eff = 1.0 / float(np.median(dt)) if dt.size else 0.0
            gap_max = float(np.max(dt)) if dt.size else 0.0
        else:
            rate_eff = gap_max = 0.0
        if rate_eff < cfg.IMU_MIN_EFFECTIVE_RATE_HZ or gap_max > cfg.IMU_MAX_GAP_S:
            imu = {k: nan for k in
                   ("imu_accel_rms", "imu_slope_rms", "imu_roll_rms",
                    "imu_pitch_rms", "imu_dominant_freq")}
            imu["_imu_quality"] = "low_rate_or_gap"
        else:
            imu["_imu_quality"] = imu.get("_imu_quality", "ok")
    else:
        imu = dict(imu_accel_rms=nan, imu_slope_rms=nan, imu_roll_rms=nan,
                   imu_pitch_rms=nan, imu_dominant_freq=nan,
                   _imu_quality="insufficient_samples")

    # ── imu_slope_rms dari model_feed wave (konsisten training physics_proxy) ──
    # build_features_v2: omega=2pi/Tp, k=omega^2/g, sigma_eta=Hs/4, slope=k*sigma_eta
    wh = mf.get("wave_height")
    wp = mf.get("wave_period")
    if not _isnan(wh) and not _isnan(wp) and wp and wp > 0:
        omega = 2.0 * math.pi / float(wp)
        k = omega ** 2 / cfg.G
        sigma_eta = float(wh) / 4.0
        imu["imu_slope_rms"] = float(k * sigma_eta)
    # else NaN -> diimpute climatology

    # ── GPS ──
    gps_lock = [s for s in gps_buf if s.valid and s.fix_quality > 0
                and not _isnan(s.latitude)]
    if mode == "station_keeping" and len(gps_lock) >= 2:
        ll = np.array([[s.latitude, s.longitude] for s in gps_lock])
        hdop = np.array([s.hdop if not _isnan(s.hdop) else 99.0
                         for s in gps_lock])
        gps = gps_drift_features(ll, hdop, mode, window_s)
    else:
        gps = gps_drift_features(None, np.array([]), mode, window_s)

    # ── pressure -> sea_level / pressure_level ──
    p_ok = [s for s in pressure_buf if s.valid and not _isnan(s.kpa)]
    if p_ok:
        mean_head = float(np.mean([s.head_m for s in p_ok if not _isnan(s.head_m)]))
        sea_level = mean_head if not _isnan(mean_head) else nan
    else:
        sea_level = nan

    # ── wind / rain ──
    wind = wind_features(wind_buf)
    rain = rain_features(rain_buf, window_s)

    # ── model_feed (gelombang/arus) ──
    wave_h = float(mf["wave_height"]) if "wave_height" in mf and mf["wave_height"] is not None else nan
    wave_p = float(mf["wave_period"]) if "wave_period" in mf and mf["wave_period"] is not None else nan
    wave_d = mf.get("wave_direction")
    cur = float(mf["current_speed"]) if "current_speed" in mf and mf["current_speed"] is not None else nan

    out = {
        # base public/real (nama PERSIS schema training)
        "sea_level": sea_level,
        "wave_height": wave_h,
        "wave_direction_sin": float(math.sin(math.radians(wave_d))) if not _isnan(wave_d) else nan,
        "wave_direction_cos": float(math.cos(math.radians(wave_d))) if not _isnan(wave_d) else nan,
        "wave_period": wave_p,
        "wind_speed": wind["wind_speed"],
        "wind_direction_sin": wind["wind_direction_sin"],
        "wind_direction_cos": wind["wind_direction_cos"],
        "rain_rate": rain["rain_rate_mmh"],
        "current_speed": cur,
        # real_sensor (excluded training, dipakai live)
        "pressure_level": sea_level,
        "imu_accel_rms": imu["imu_accel_rms"],
        "imu_slope_rms": imu["imu_slope_rms"],
        "imu_roll_rms": imu["imu_roll_rms"],
        "imu_pitch_rms": imu["imu_pitch_rms"],
        "imu_dominant_freq": imu["imu_dominant_freq"],
        "gps_drift_speed": gps["gps_drift_speed"],
        "gps_drift_distance": gps["gps_drift_distance"],
        "gps_heading_change": gps["gps_heading_change"],
        # flags (tidak masuk model)
        "_imu_quality": imu["_imu_quality"],
        "_gps_low_confidence": gps["_gps_low_confidence"],
        "_wind_gust_ms": wind.get("wind_gust_ms", nan),
        "_rain_mm": rain["rain_mm"],
        "_pressure_has_tide": (p_ok is not None and len(p_ok) > 0),  # TODO detide
        "_ntp_synced": bool(imu_ok and imu_ok[0].ntp_synced) if imu_ok else False,
    }
    return out
