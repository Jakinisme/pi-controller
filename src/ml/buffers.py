"""Ring buffer thread-safe per sensor.

Kapasitas dalam detik. Tidak boleh memblokir thread pembaca sensor.
Fan-out read-only: consumer (ML) baca snapshot, tidak ubah buffer.
"""
import threading
import time
from collections import deque
from typing import Deque, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer:
    """Ring buffer per sensor dengan kapasitas berbasis detik monotonic.

    Penyimpanan: (timestamp_mono, sample). Append non-blocking (lock
    sangat singkat). Trim otomatis berdasarkan window_seconds.
    """

    __slots__ = ("_buf", "_lock", "_window_s", "_maxlen", "_name")

    def __init__(self, name: str, window_seconds: float, max_rate_hz: float = 50.0):
        self._name = name
        self._window_s = float(window_seconds)
        # maxlen hard cap mencegah memory blow kalau consumer lambat.
        self._maxlen = int(max_rate_hz * window_seconds * 1.2) + 16
        self._buf: Deque = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    def append(self, sample) -> None:
        """Append non-blocking. sample harus punya .timestamp_mono."""
        with self._lock:
            self._buf.append(sample)
            # trim by time (deque already caps by count as backstop)
            ts = sample.timestamp_mono
            cutoff = ts - self._window_s
            buf = self._buf
            while len(buf) > 1 and buf[0].timestamp_mono < cutoff:
                buf.popleft()

    def snapshot(self) -> List:
        """Copy seluruh buffer (read-only). Tidak memblokir writer lama."""
        with self._lock:
            return list(self._buf)

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def window_span_s(self) -> float:
        """Span waktu sampel tertua-terbaru. 0 jika <2 sampel."""
        with self._lock:
            if len(self._buf) < 2:
                return 0.0
            return self._buf[-1].timestamp_mono - self._buf[0].timestamp_mono

    def largest_gap_s(self) -> float:
        """Gap terbesar antar sampel berurutan. Untuk cek kualitas."""
        with self._lock:
            if len(self._buf) < 2:
                return 0.0
            it = iter(self._buf)
            prev = next(it)
            mx = 0.0
            for s in it:
                g = s.timestamp_mono - prev.timestamp_mono
                if g > mx:
                    mx = g
                prev = s
            return mx
