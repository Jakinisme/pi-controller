"""Tipping bucket driver — reed switch pulse via pigpio callback.

Ref: E:\\Zaky-Sekolah\\UGM\\src\\flood-sensorr\\flood-sensor\\src\\curah_hujan.cpp
- FALLING edge, INPUT_PULLUP.
- MILIMETER_PER_TIP = 0.70 (ref; TODO verifikasi fisik bucket ini).
- Debounce 500ms (ref delay(500) blocking). Di Pi: pigpio glitch filter
  500000us, JANGAN delay blocking.

Status: STUB. pin None -> no-op.
"""
import time
import threading
from typing import Optional

from .. import config_ml as cfg
from ..sensor_contract import RainSample, now_mono, now_wall


class TippingBucketReader:
    """Pembaca tipping bucket. Satu instance. Thread-safe via callback."""

    def __init__(self, pi=None, gpio=None):
        self._pi = pi
        self._gpio = gpio or cfg.TIPPING_BUCKET_GPIO
        self._running = False
        self._pulse_cb = None
        self._tips = 0
        self._lock = threading.Lock()
        self._last_tip_mono = 0.0

    @property
    def ready(self) -> bool:
        return self._pi is not None and self._gpio is not None

    def _on_tip(self, gpio, level, tick):
        """pigpio callback. glitch filter menangani debounce 500ms."""
        with self._lock:
            self._tips += 1
            self._last_tip_mono = now_mono()

    def start(self):
        if self._running or not self.ready:
            return
        self._running = True
        # glitch filter debounce (pigpio: micros). 500ms = 500000us
        self._pi.set_glitch_filter(self._gpio, cfg.TIPPING_DEBOUNCE_US)
        try:
            import pigpio
            self._pulse_cb = self._pi.callback(self._gpio, pigpio.FALLING, self._on_tip)
        except ImportError:
            self._pulse_cb = None

    def read_sample(self) -> Optional[RainSample]:
        if not self.ready:
            return None
        with self._lock:
            tips = self._tips
        rain_mm = tips * cfg.RAIN_MM_PER_TIP
        return RainSample(
            timestamp_mono=now_mono(), timestamp_wall=now_wall(),
            ntp_synced=False, valid=tips > 0, tips=tips, rain_mm=rain_mm,
        )

    def stop(self):
        self._running = False
        if self._pulse_cb is not None:
            self._pulse_cb.cancel()
            self._pulse_cb = None