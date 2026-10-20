"""Data logger — JSONL dengan rotasi ukuran disk.

Log fitur + ringkasan data mentah per AGG_INTERVAL. provenance="real_sensor".
Bahan retraining dengan data asli nanti. SD card terbatas -> rotasi.
"""
import json
import os
import time
import threading
from typing import Optional

from . import config_ml as cfg


class DataLogger:
    """JSONL logger dengan rotasi ukuran. Thread-safe (lock singkat)."""

    def __init__(self, log_dir: Optional[str] = None,
                 max_bytes: Optional[int] = None, keep: Optional[int] = None):
        self.log_dir = os.path.expanduser(log_dir or str(cfg.LOG_DIR))
        self.max_bytes = max_bytes or cfg.LOG_MAX_BYTES
        self.keep = keep or cfg.LOG_KEEP
        self._lock = threading.Lock()
        os.makedirs(self.log_dir, exist_ok=True)
        self._path = os.path.join(self.log_dir, "ml_features.jsonl")

    def _rotate_if_needed(self):
        try:
            if os.path.getsize(self._path) >= self.max_bytes:
                # rotate: .jsonl -> .1 -> .2 ...
                for i in range(self.keep - 1, 0, -1):
                    src = f"{self._path}.{i}"
                    dst = f"{self._path}.{i + 1}"
                    if os.path.exists(src):
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.rename(src, dst)
                if os.path.exists(self._path):
                    os.rename(self._path, f"{self._path}.1")
        except OSError:
            pass

    def log(self, result: dict, provenance: str = "real_sensor", mode: str = ""):
        """Tulis satu baris JSONL. Non-blocking best-effort (swallow IO error)."""
        entry = {
            "ts_wall": result.get("ts_wall"),
            "ts_mono": result.get("ts_mono"),
            "provenance": provenance,
            "mode": mode,
            "model_version": result.get("model_version"),
            "features": result.get("features"),
            "feature_quality": result.get("feature_quality"),
            "degraded": result.get("degraded"),
            "missing_sensors": result.get("missing_sensors"),
            "score": result.get("score"),
            "is_anomaly": result.get("is_anomaly"),
            "interpretation": result.get("interpretation"),
            "storm_context": result.get("storm_context"),
            "ood": result.get("input_out_of_distribution"),
            "scoring_active": result.get("scoring_active"),
            "score_reason": result.get("score_reason"),
            "notes": result.get("notes"),
        }
        try:
            with self._lock:
                self._rotate_if_needed()
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass   # SD card penuh/error: jangan crash kontrol loop

    def replay(self):
        """Iterator baris JSONL (untuk mode replay)."""
        with self._lock:
            if not os.path.exists(self._path):
                return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
