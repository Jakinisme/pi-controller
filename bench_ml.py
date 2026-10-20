#!/usr/bin/env python3
"""Skrip ukur waktu per siklus ML — jalankan di Pi.

Ukuran: fan-out + compute_base + cycle. Laporan: N siklus, mean/min/max.
"""
import sys, os, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_mock import install
install()

from src.ml.buffers import RingBuffer
from src.ml.feature_builder import compute_base
from src.ml.ml_runner import MLRunner
from src.ml.synthetic import (
    synthetic_imu_stream, synthetic_gps_stream, synthetic_pressure_stream,
    synthetic_wind_stream, synthetic_rain_stream,
)

def fill(buf, stream):
    for s in stream:
        buf.append(s)

def main(n=20):
    imu_buf = RingBuffer("imu", 130, 50.0)
    gps_buf = RingBuffer("gps", 130, 1.0)
    p_buf = RingBuffer("pressure", 130, 1.0)
    w_buf = RingBuffer("wind", 130, 1.0)
    r_buf = RingBuffer("rain", 130, 1.0)
    fill(imu_buf, synthetic_imu_stream(fs=10.0, freq_hz=0.15, duration_s=125))
    fill(gps_buf, synthetic_gps_stream(duration_s=125, fs=1.0))
    fill(p_buf, synthetic_pressure_stream(head_m=0.3, duration_s=125))
    fill(w_buf, synthetic_wind_stream(speed_ms=6.0, duration_s=125))
    fill(r_buf, synthetic_rain_stream(rain_mmh=0.5, duration_s=125))

    runner = MLRunner()
    # warmup (load model path, tidak di-cache kalau scoring off)
    base = compute_base(imu_buf.snapshot(), gps_buf.snapshot(), p_buf.snapshot(),
                         w_buf.snapshot(), r_buf.snapshot(), model_feed=None,
                         mode="station_keeping", window_s=86400)
    runner.cycle(base, mode="station_keeping", month=1)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        base = compute_base(imu_buf.snapshot(), gps_buf.snapshot(), p_buf.snapshot(),
                             w_buf.snapshot(), r_buf.snapshot(), model_feed=None,
                             mode="station_keeping", window_s=86400)
        runner.cycle(base, mode="station_keeping", month=1)
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"ML cycle timing ({n} runs, ms):")
    print(f"  mean: {statistics.mean(times):.2f}")
    print(f"  min:  {min(times):.2f}")
    print(f"  max:  {max(times):.2f}")
    print(f"  stdev: {statistics.stdev(times):.2f}")
    print(f"  compute_base+cycle: {statistics.mean(times):.2f} ms/siklus")
    print(f"  AGG_INTERVAL={runner.agg_window_s}s -> overhead {statistics.mean(times)/runner.agg_window_s*100:.4f}%")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)