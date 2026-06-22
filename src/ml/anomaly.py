"""
Anomaly Detection using Isolation Forest.

Monitors sensor data for anomalies:
  - GPS position jumps
  - IMU spikes
  - Magnetometer disturbances (EMI)

Flags anomalous readings so the fusion layer can downweight or reject them.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

from src.config import ML_ANOMALY_HISTORY_SIZE, ML_MODEL_PATH
from src.utils.logger import setup_logger

log = setup_logger("anomaly")

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None
    log.warning("scikit-learn not installed; anomaly detection disabled")


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool = False
    anomaly_score: float = 0.0    # Negative = more anomalous
    gps_anomaly: bool = False
    imu_anomaly: bool = False
    mag_anomaly: bool = False
    timestamp: float = 0.0


class AnomalyDetector:
    """Isolation Forest-based sensor anomaly detector."""

    def __init__(self, history_size: int = ML_ANOMALY_HISTORY_SIZE):
        self._history_size = history_size
        self._model: Optional[IsolationForest] = None
        self._is_fitted = False

        # Feature history buffer: [gps_speed, gps_lat_change, gps_lon_change,
        #                           gyro_z, accel_magnitude, mag_heading_change]
        self._history: deque = deque(maxlen=history_size)
        self.result = AnomalyResult()

        # Simple thresholds for immediate flagging (before model is fitted)
        self._gps_speed_max = 5.0         # m/s, unlikely for small ASV
        self._gps_jump_max = 0.0005       # degrees per reading (~50m)
        self._gyro_z_max = 90.0           # deg/s, unlikely for surface vehicle
        self._accel_mag_range = (7.0, 13.0)  # m/s^2 (around 9.81g)

        self._prev_lat = 0.0
        self._prev_lon = 0.0
        self._prev_mag_heading = 0.0

    def _build_model(self):
        """Initialize the Isolation Forest model."""
        if IsolationForest is None:
            return
        self._model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # Expect ~5% anomalies
            max_samples="auto",
            random_state=42,
        )

    def add_sample(
        self,
        gps_speed: float,
        lat: float,
        lon: float,
        gyro_z: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        mag_heading: float,
    ):
        """Add a sensor sample to the history buffer."""
        lat_change = abs(lat - self._prev_lat) if self._prev_lat != 0 else 0.0
        lon_change = abs(lon - self._prev_lon) if self._prev_lon != 0 else 0.0
        heading_change = abs(mag_heading - self._prev_mag_heading)
        if heading_change > 180:
            heading_change = 360 - heading_change
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        features = [
            gps_speed,
            lat_change,
            lon_change,
            abs(gyro_z),
            accel_mag,
            heading_change,
        ]
        self._history.append(features)

        self._prev_lat = lat
        self._prev_lon = lon
        self._prev_mag_heading = mag_heading

    def fit(self):
        """Train the Isolation Forest on collected history.
        Call this after collecting enough normal data (e.g., 200+ samples).
        """
        if IsolationForest is None:
            log.warning("sklearn not available, cannot fit model")
            return

        if len(self._history) < 50:
            log.warning("Not enough data to fit (%d samples)", len(self._history))
            return

        if self._model is None:
            self._build_model()

        data = np.array(self._history)
        self._model.fit(data)
        self._is_fitted = True
        log.info("Anomaly model fitted on %d samples", len(self._history))

    def detect(
        self,
        gps_speed: float,
        lat: float,
        lon: float,
        gyro_z: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        mag_heading: float,
    ) -> AnomalyResult:
        """Run anomaly detection on current sensor readings.
        
        Returns:
            AnomalyResult indicating which sensors are anomalous.
        """
        now = time.time()

        # Threshold-based detection (always active)
        lat_change = abs(lat - self._prev_lat) if self._prev_lat != 0 else 0.0
        lon_change = abs(lon - self._prev_lon) if self._prev_lon != 0 else 0.0
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        heading_change = abs(mag_heading - self._prev_mag_heading)
        if heading_change > 180:
            heading_change = 360 - heading_change

        gps_anomaly = (
            gps_speed > self._gps_speed_max
            or lat_change > self._gps_jump_max
            or lon_change > self._gps_jump_max
        )
        imu_anomaly = (
            abs(gyro_z) > self._gyro_z_max
            or accel_mag < self._accel_mag_range[0]
            or accel_mag > self._accel_mag_range[1]
        )
        mag_anomaly = heading_change > 30  # >30 deg jump in one reading

        # ML-based detection (if model is fitted)
        ml_anomaly = False
        ml_score = 0.0
        if self._is_fitted and self._model is not None:
            features = np.array([[
                gps_speed, lat_change, lon_change,
                abs(gyro_z), accel_mag, heading_change,
            ]])
            prediction = self._model.predict(features)[0]  # -1 = anomaly, 1 = normal
            score = self._model.score_samples(features)[0]
            ml_anomaly = prediction == -1
            ml_score = float(score)

        is_anomaly = gps_anomaly or imu_anomaly or mag_anomaly or ml_anomaly

        self.result = AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=ml_score,
            gps_anomaly=gps_anomaly,
            imu_anomaly=imu_anomaly,
            mag_anomaly=mag_anomaly,
            timestamp=now,
        )

        if is_anomaly:
            log.warning(
                "Anomaly detected: GPS=%s IMU=%s MAG=%s ML=%s",
                gps_anomaly, imu_anomaly, mag_anomaly, ml_anomaly,
            )

        # Add to history (even anomalous data, for retraining)
        self.add_sample(gps_speed, lat, lon, gyro_z, accel_x, accel_y, accel_z, mag_heading)

        return self.result

    def save_model(self, path: str = None):
        """Save the fitted model to disk."""
        if not self._is_fitted or self._model is None:
            log.warning("No fitted model to save")
            return
        import pickle
        import os
        save_path = path or os.path.join(ML_MODEL_PATH, "anomaly_model.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self._model, f)
        log.info("Anomaly model saved to %s", save_path)

    def load_model(self, path: str = None):
        """Load a pre-trained model from disk."""
        if IsolationForest is None:
            return
        import pickle
        import os
        load_path = path or os.path.join(ML_MODEL_PATH, "anomaly_model.pkl")
        if not os.path.exists(load_path):
            log.warning("Model file not found: %s", load_path)
            return
        with open(load_path, "rb") as f:
            self._model = pickle.load(f)
        self._is_fitted = True
        log.info("Anomaly model loaded from %s", load_path)
