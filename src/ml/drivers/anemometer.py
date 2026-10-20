"""Anemometer driver — reed/Hall pulse count via pigpio callback.

Ref: E:\\Zaky-Sekolah\\UGM\\src\\flood-sensorr\\flood-sensor\\src\\anemometer.cpp
- RISING pulse, debounce 5ms (micros diff di ref; di Pi pakai glitch filter).
- RPS = count / window (ref 10s).
- m/s = -0.0181*RPS^2 + 1.3859*RPS + 1.4055 (ref).
- deadband <=1.5 m/s -> 0 (ref).
- TODO: level shifter kalau sensor 5V (Pi 3.3V). Pin Pi belum set.

Status: STUB. pin None -> no-op.
"""
import time
import math
import threading
from typing import Optional

from .. import config_ml as cfg
from ..sensor_contract import WindSample, now_mono, now_wall


class AnemometerReader:
    """Pembaca anemometer pulse. Satu instance. Thread-safe via callback."""

    def __init__(self, pi=None, gpio=None):
        self._pi = pi
        self._gpio = gpio or cfg.ANEMOMETER_GPIO
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pulse_cb = None
        self._count = 0
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._pi is not None and self._gpio is not None

    def _on_pulse(self, gpio, level, tick):
        """pigpio callback. glitch filter menangani debounce."""
        with self._lock:
            self._count += 1

    def start(self):
        if self._running or not self.ready:
            return
        self._running = True
        # glitch filter: abaikan pulsa < ANEMOMETER_DEBOUNCE_US
        self._pi.set_glitch_filter(self._gpio, cfg.ANEMOMETER_DEBOUNCE_US)
        try:
            import pigpio
            self._pulse_cb = self._pi.callback(self._gpio, pigpio.RISING, self._on_pulse)
        except ImportError:
            self._pulse_cb = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            time.sleep(cfg.ANEMOMETER_WINDOW_S)
            with self._lock:
                count = self._count
                self._count = 0
            rps = count / cfg.ANEMOMETER_WINDOW_S
            ms = (cfg.ANEMOMETER_COEF_A * rps * rps
                  + cfg.ANEMOMETER_COEF_B * rps
                  + cfg.ANEMOMETER_COEF_C)
            if ms <= cfg.ANEMOMETER_DEADBAND_MS:
                ms = 0.0
            # sample disimpan caller via poll; di sini cuma update state
            self._last = WindSample(
                timestamp_mono=now_mono(), timestamp_wall=now_wall(),
                ntp_synced=False, valid=True, rps=rps, speed_ms=ms,
            )

    def read_sample(self) -> Optional[WindSample]:
        return getattr(self, "_last", None)

    def stop(self):
        self._running = False
        if self._pulse_cb is not None:
            self._pulse_cb.cancel()
            self._pulse_cb = None