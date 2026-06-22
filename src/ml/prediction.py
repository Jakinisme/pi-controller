"""
Drift Prediction using Random Forest.

Predicts environmental effects (wind/current drift) based on historical data.
Can be used to pre-compensate thrust commands for better station-keeping.

Features: heading, speed, time_of_day, wind_direction (if available)
Target: drift_velocity_north, drift_velocity_east
"""

import time
import os
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

from src.config import ML_MODEL_PATH
from src.utils.logger import setup_logger

log = setup_logger("predict")

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None
    log.warning("scikit-learn not installed; drift prediction disabled")


@dataclass
class DriftPrediction:
    """Predicted drift from environmental forces."""
    drift_north: float = 0.0    # m/s predicted northward drift
    drift_east: float = 0.0     # m/s predicted eastward drift
    confidence: float = 0.0     # Model confidence [0, 1]
    timestamp: float = 0.0
    valid: bool = False


class DriftPredictor:
    """Random Forest-based drift predictor."""

    def __init__(self, history_size: int = 1000):
        self._history_size = history_size
        self._model: Optional[RandomForestRegressor] = None
        self._is_fitted = False
        self._history: deque = deque(maxlen=history_size)
        self._targets: deque = deque(maxlen=history_size)
        self.prediction = DriftPrediction()

    def add_training_sample(
        self,
        heading: float,
        speed: float,
        surge: float,
        sway: float,
        yaw: float,
        # Observed drift (computed by comparing commanded vs actual movement)
        observed_drift_north: float,
        observed_drift_east: float,
    ):
        """Add a training sample.
        
        Drift is estimated by comparing: what the thrusters should produce
        vs what actually happened (GPS movement minus expected thrust).
        """
        hour = (time.time() % 86400) / 3600  # Hour of day (0-24)
        features = [
            np.sin(np.radians(heading)),
            np.cos(np.radians(heading)),
            speed,
            surge,
            sway,
            yaw,
            hour / 24.0,
        ]
        self._history.append(features)
        self._targets.append([observed_drift_north, observed_drift_east])

    def fit(self):
        """Train the Random Forest model on collected data."""
        if RandomForestRegressor is None:
            log.warning("sklearn not available, cannot fit model")
            return

        if len(self._history) < 100:
            log.warning("Not enough data to fit (%d samples)", len(self._history))
            return

        X = np.array(self._history)
        y = np.array(self._targets)

        self._model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
        )
        self._model.fit(X, y)
        self._is_fitted = True

        # Compute training score
        score = self._model.score(X, y)
        log.info("Drift model fitted on %d samples, R2=%.3f", len(self._history), score)

    def predict(self, heading: float, speed: float, surge: float, sway: float, yaw: float) -> DriftPrediction:
        """Predict current drift based on vehicle state.
        
        Returns:
            DriftPrediction with estimated drift velocities.
        """
        now = time.time()

        if not self._is_fitted or self._model is None:
            return DriftPrediction(timestamp=now, valid=False)

        hour = (time.time() % 86400) / 3600
        features = np.array([[
            np.sin(np.radians(heading)),
            np.cos(np.radians(heading)),
            speed,
            surge,
            sway,
            yaw,
            hour / 24.0,
        ]])

        try:
            pred = self._model.predict(features)[0]
            # Confidence based on prediction variance (if available)
            tree_preds = np.array([tree.predict(features)[0] for tree in self._model.estimators_])
            variance = np.mean(np.var(tree_preds, axis=0))
            confidence = max(0.0, 1.0 - variance)

            self.prediction = DriftPrediction(
                drift_north=float(pred[0]),
                drift_east=float(pred[1]),
                confidence=float(confidence),
                timestamp=now,
                valid=True,
            )
        except Exception as e:
            log.error("Prediction error: %s", e)
            self.prediction = DriftPrediction(timestamp=now, valid=False)

        return self.prediction

    def save_model(self, path: str = None):
        """Save the fitted model to disk."""
        if not self._is_fitted or self._model is None:
            log.warning("No fitted model to save")
            return
        import pickle
        save_path = path or os.path.join(ML_MODEL_PATH, "drift_model.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self._model, f)
        log.info("Drift model saved to %s", save_path)

    def load_model(self, path: str = None):
        """Load a pre-trained model from disk."""
        if RandomForestRegressor is None:
            return
        import pickle
        load_path = path or os.path.join(ML_MODEL_PATH, "drift_model.pkl")
        if not os.path.exists(load_path):
            log.warning("Model file not found: %s", load_path)
            return
        with open(load_path, "rb") as f:
            self._model = pickle.load(f)
        self._is_fitted = True
        log.info("Drift model loaded from %s", load_path)
