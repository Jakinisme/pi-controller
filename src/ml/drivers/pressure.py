"""HX710B pressure driver — DOUT + SCK bit-bang via pigpio.

Protocol (VERIFIKASI ke datasheet HX710B, jangan dari ingatan):
- 24-bit two's complement, DOUT low = data siap.
- Jumlah pulsa SCK menentukan mode/rate.
- SCK high > 60 us (verifikasi datasheet) -> chip power-down.
  Bit-bang Python di Pi punya jitter -> pakai pigpio waves/ticks.

Status: STUB. pin None -> no-op. Kalibrasi offset/scale TODO verifikasi fisik.
Kalau failure rate tinggi saat bench test -> usulkan ESP32 baca + JSON serial.
"""
import time
import math
import threading
from typing import Optional

from .. import config_ml as cfg
from ..sensor_contract import PressureSample, now_mono, now_wall


_SAT_MAX = 0x7FFFFF   # 24-bit two's complement positive max
_ALL_ONES = 0xFFFFFF


class HX710BReader:
    """Pembaca HX710B. Satu instance per sensor. Thread-safe read."""

    def __init__(self, pi=None, dout_gpio=None, sck_gpio=None):
        self._pi = pi   # pigpio.pi() instance (None di mock/test)
        self._dout = dout_gpio or cfg.HX710B_DOUT_GPIO
        self._sck = sck_gpio or cfg.HX710B_SCK_GPIO
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # median filter
        self._recent = []

    @property
    def ready(self) -> bool:
        return self._pi is not None and self._dout is not None and self._sck is not None

    def _read_counts(self) -> Optional[int]:
        """Baca 24-bit. Return None kalau timeout/saturasi/invalid.
        TODO: verifikasi timing SCK dari datasheet."""
        if not self.ready:
            return None
        # tunggu DOUT low (data ready) — timeout 100ms
        t0 = time.monotonic()
        while self._pi.read(self._dout) == 1:
            if time.monotonic() - t0 > 0.1:
                return None
            time.sleep(0.001)
        counts = 0
        for _ in range(24):
            self._pi.write(self._sck, 1)
            time.sleep(0.000001)   # 1us (jaga < 60us; verifikasi datasheet)
            self._pi.write(self._sck, 0)
            time.sleep(0.000001)
            counts = (counts << 1) | self._pi.read(self._dout)
        # pulsa tambahan untuk mode (verifikasi datasheet: 25/26/27 pulsa)
        # default 25 (gain 128 ch A). TODO konfirmasi.
        self._pi.write(self._sck, 1); time.sleep(0.000001)
        self._pi.write(self._sck, 0)
        # two's complement sign extend
        if counts & 0x800000:
            counts -= 0x1000000
        # reject saturation / all-ones
        if counts >= _SAT_MAX or counts <= -_SAT_MAX or (counts & 0xFFFFFF) == _ALL_ONES:
            return None
        return counts

    def _counts_to_kpa(self, counts: int) -> float:
        """offset + scale (TODO kalibrasi 2-titik fisik)."""
        return (counts - cfg.HX710B_OFFSET_COUNTS) * cfg.HX710B_SCALE_KPA_PER_COUNT

    def read_sample(self) -> Optional[PressureSample]:
        """Satu bacaan median-filtered. None kalau tidak ada/gagal."""
        if not self.ready:
            return None
        c = self._read_counts()
        if c is None:
            return None
        self._recent.append(c)
        if len(self._recent) > 5:
            self._recent = self._recent[-5:]
        med = sorted(self._recent)[len(self._recent) // 2]
        kpa = self._counts_to_kpa(med)
        head = (kpa * 1000.0 - cfg.P_ATM) / (cfg.RHO_SEAWATER * cfg.G) - cfg.TUBE_DEPTH_OFFSET_M
        if abs(head) > cfg.HX710B_MAX_HEAD_M:
            return None
        return PressureSample(
            timestamp_mono=now_mono(), timestamp_wall=now_wall(),
            ntp_synced=False, valid=True, counts=med, kpa=kpa, head_m=head,
        )

    def start(self):
        if self._running or not self.ready:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        interval = 1.0 / cfg.HX710B_SAMPLE_RATE_HZ
        while self._running:
            t0 = time.monotonic()
            s = self.read_sample()
            # caller (MLFanout) polls; tidak attach buffer langsung
            elapsed = time.monotonic() - t0
            sl = max(0.0, interval - elapsed)
            if sl > 0:
                time.sleep(sl)

    def stop(self):
        self._running = False