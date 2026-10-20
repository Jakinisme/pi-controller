"""ML runner — load artifacts, rolling derived, shadow scoring (gated OFF).

Port dari machine-learning/src/v2/scorer_v2.py. Scoring mati permanen
(SCORING_ENABLED=False) sampai model sensor-trained schema-compatible ada.
Ekstraksi fitur + logging + telemetri tetap jalan (shadow mode).

Alur:
  runner = MLRunner()
  base = compute_base(...)             # dari feature_builder
  out = runner.cycle(base, mode, month)  # derived + (scoring off) + telemetry payload
"""
import json
import math
import os
import time
from collections import deque
from typing import Optional

import numpy as np

from . import config_ml as cfg
from .data_logger import DataLogger


def _isnan(v) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


# fitur base yang jadi rolling history untuk derived (konsisten build_features_v2)
BASE_FOR_DERIVED = ["sea_level", "wave_height", "wind_speed", "rain_rate", "current_speed"]
DERIVED_WITH_RATE = ["sea_level", "wave_height", "wind_speed"]


class MLRunner:
    """Load artifacts sekali, proses cycle per AGG_INTERVAL.

    Saat start: assert nama+urutan fitur live == schema == kolom model.
    Kalau tidak cocok -> scoring mati, fitur+logging tetap jalan.
    """

    def __init__(self, artifacts_dir: Optional[str] = None,
                 logger: Optional[DataLogger] = None):
        d = artifacts_dir or str(cfg.ARTIFACTS_DIR)
        self.schema_path = os.path.join(d, "feature_schema.json")
        self.flist_path = os.path.join(d, "feature_list.json")
        self.clim_path = os.path.join(d, "climatology.json")
        self.model_path = str(cfg.MODEL_PATH)

        self._load_artifacts()
        self.history = deque(maxlen=self.rolling_window)
        self.logger = logger or DataLogger()
        self._scoring_active = False
        self._score_reason = "scoring_disabled_by_default"
        self._init_scoring_guard()

    # ── artifact loading ──────────────────────────────
    def _load_artifacts(self):
        with open(self.schema_path) as f:
            self.schema = json.load(f)
        with open(self.flist_path) as f:
            self.flist = json.load(f)
        with open(self.clim_path) as f:
            self.climatology = json.load(f)
        self.features = self.flist["features"]          # canonical order
        self.agg_window_s = int(self.flist.get("agg_window_seconds",
                                               cfg.AGG_WINDOW_SECONDS))
        self.rolling_window = int(self.flist.get("rolling_window",
                                                 cfg.ROLLING_WINDOW))
        self.units = self.schema.get("units", {})

    def _init_scoring_guard(self):
        """Cek apakah scoring boleh aktif. Default OFF (aturan 6.6)."""
        if not cfg.SCORING_ENABLED:
            self._scoring_active = False
            self._score_reason = "SCORING_ENABLED=False (aturan 6.6: model proxy)"
            return
        # kalau di-enable di config, masih cek model load + schema match
        try:
            import joblib
            model = joblib.load(self.model_path)
            got = list(model.feature_names_in_)
            if got != self.features:
                self._scoring_active = False
                self._score_reason = f"model feature mismatch: {got} != {self.features}"
                return
            self.model = model
            self._scoring_active = True
            self._score_reason = "ok"
        except Exception as e:
            self._scoring_active = False
            self._score_reason = f"model_load_failed: {e}"

    # ── derived (konsisten build_features_v2 / scorer_v2) ──
    def _derived(self, base: dict, month: int) -> dict:
        out = dict(base)
        for col in BASE_FOR_DERIVED:
            series = [h.get(col) for h in self.history if not _isnan(h.get(col))]
            cur = base.get(col)
            if not _isnan(cur):
                series.append(cur)
            if series:
                ma = float(np.mean(series))
                std = float(np.std(series)) if len(series) >= 2 else 0.0
            else:
                ma = std = float("nan")
            out[f"{col}_ma"] = ma
            if f"{col}_std" in self.features:
                out[f"{col}_std"] = std
        prev = self.history[-1] if self.history else None
        for col in DERIVED_WITH_RATE:
            if prev is None or _isnan(base.get(col)) or _isnan(prev.get(col)):
                out[f"{col}_delta"] = float("nan")
                out[f"{col}_rate"] = float("nan")
            else:
                d = base[col] - prev[col]
                out[f"{col}_delta"] = float(d)
                out[f"{col}_rate"] = float(d / self.agg_window_s)
        for col in ["sea_level", "wave_height", "wind_speed"]:
            clim = self.climatology.get(col, {}).get(str(month))
            if clim and clim["std"] and not _isnan(base.get(col)):
                out[f"{col}_deseasonalized"] = (base[col] - clim["mean"]) / clim["std"]
            else:
                out[f"{col}_deseasonalized"] = float("nan")
        rain_series = [h.get("rain_rate") for h in list(self.history)[-2:]] + [base.get("rain_rate")]
        rain_series = [r * 24.0 for r in rain_series if not _isnan(r)]
        out["rain_acc_3d"] = float(sum(rain_series)) if rain_series else float("nan")
        return out

    def _missing_sensors(self, base: dict, mode: str):
        m = set()
        if _isnan(base.get("sea_level")):
            m.add("pressure")
        if _isnan(base.get("imu_accel_rms")) or _isnan(base.get("imu_dominant_freq")):
            m.add("imu")
        if mode == "station_keeping" and _isnan(base.get("gps_drift_speed")):
            m.add("gps")
        if _isnan(base.get("wind_speed")):
            m.add("wind")
        if _isnan(base.get("rain_rate")):
            m.add("rain")
        if _isnan(base.get("wave_height")):
            m.add("wave_feed")
        if _isnan(base.get("current_speed")):
            m.add("currents")
        missing = sorted(m)
        core_missing = "pressure" in m or "imu" in m
        return missing, core_missing

    def _impute(self, derived: dict, month: int) -> dict:
        filled = dict(derived)
        for feat in self.features:
            v = filled.get(feat)
            if _isnan(v):
                clim = self.climatology.get(feat, {}).get(str(month))
                if clim and clim["mean"] is not None:
                    filled[feat] = clim["mean"]
                else:
                    filled[feat] = 0.0
        return filled

    # ── out-of-distribution guard ─────────────────────
    def _check_ood(self, filled: dict) -> bool:
        """Bandingkan fitur live dengan rentang climatology. Di luar jauh -> OOD."""
        for feat in self.features:
            clim = self.climatology.get(feat)
            if not clim:
                continue
            v = filled.get(feat)
            if _isnan(v):
                continue
            # pakai bulan saat ini
            month = time.gmtime().tm_mon
            c = clim.get(str(month))
            if not c or not c.get("std"):
                continue
            z = abs(v - c["mean"]) / c["std"]
            if z > 6.0:   # >6 sigma = OOD
                return True
        return False

    # ── main cycle ────────────────────────────────────
    def cycle(self, base: dict, mode: str = "station_keeping",
              month: Optional[int] = None) -> dict:
        if month is None:
            month = time.gmtime().tm_mon
        derived = self._derived(base, month)
        missing, core_missing = self._missing_sensors(base, mode)
        filled = self._impute(derived, month)

        wind = base.get("wind_speed")
        rain = base.get("rain_rate")
        storm = ((not _isnan(wind) and wind >= cfg.WIND_STORM_THRESHOLD)
                 or (not _isnan(rain) and rain >= cfg.RAIN_HEAVY_THRESHOLD))

        notes = []
        if base.get("_pressure_has_tide"):
            notes.append("pressure_level mengandung pasang surut - TODO detide")
        if not cfg.MODEL_FEED_ENABLED:
            notes.append("model_feed=None (stub): fitur gelombang/arus diimpute climatology")

        # scoring OFF by default
        score = None
        is_anomaly = None
        interp = "shadow_no_scoring"
        if self._scoring_active and not core_missing:
            try:
                import pandas as pd
                x = pd.DataFrame([[filled[f] for f in self.features]],
                                 columns=self.features)
                score = float(self.model.decision_function(x)[0])
                is_anomaly = bool(self.model.predict(x)[0] == -1)
                if not is_anomaly:
                    interp = "normal"
                elif storm:
                    interp = "anomali_dijelaskan_badai"
                else:
                    interp = "anomali_cuaca_tenang"
            except Exception as e:
                self._scoring_active = False
                self._score_reason = f"score_runtime_error: {e}"
                notes.append(self._score_reason)
        elif core_missing:
            interp = "sensor_core_hilang"

        ood = self._check_ood(filled)

        result = {
            "ts_wall": time.time(),
            "ts_mono": time.monotonic(),
            "features": {f: float(filled[f]) for f in self.features},
            "feature_quality": {
                "imu": base.get("_imu_quality", "unknown"),
                "gps_low_confidence": base.get("_gps_low_confidence", False),
                "ntp_synced": base.get("_ntp_synced", False),
            },
            "degraded": len(missing) > 0,
            "missing_sensors": missing,
            "score": score,
            "is_anomaly": is_anomaly,
            "storm_context": bool(storm),
            "interpretation": interp,
            "model_version": self.flist.get("model_version", "v2"),
            "input_out_of_distribution": ood,
            "non_operational": cfg.SHADOW_MODE,
            "scoring_active": self._scoring_active,
            "score_reason": self._score_reason,
            "notes": notes,
        }

        # update history setelah derived (delta pakai window sebelumnya)
        self.history.append(base)
        # log + telemetry payload
        self.logger.log(result, provenance="real_sensor", mode=mode)
        return result

    def telemetry_payload(self, result: dict) -> dict:
        """Payload untuk firebase push_ml (node telemetry/ml)."""
        return {
            "score": result["score"],
            "is_anomaly": result["is_anomaly"],
            "interpretation": result["interpretation"],
            "degraded": result["degraded"],
            "missing_sensors": result["missing_sensors"],
            "storm_context": result["storm_context"],
            "ood": result["input_out_of_distribution"],
            "non_operational": result["non_operational"],
            "scoring_active": result["scoring_active"],
            "model_version": result["model_version"],
            "ts": round(result["ts_wall"], 3),
        }
