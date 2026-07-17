#!/usr/bin/env python3
"""
Magnetometer Calibration Tool for ASV.

Usage:
    python calibrate_mag.py              # Full calibration (rotate buoy for 30s)
    python calibrate_mag.py --dump       # Dump raw X/Y/Z readings for 10s
    python calibrate_mag.py --test       # Test current calibration (show heading)
    python calibrate_mag.py --duration 60  # Calibrate for 60 seconds

During calibration:
    1. Run this script
    2. SLOWLY rotate the buoy 360° in the horizontal plane (yaw)
    3. Try to keep it level — tilt causes errors
    4. The script will compute and save hard-iron offsets

Offsets are saved to ~/pi-controller/mag_calibration.json
and auto-loaded by the magnetometer driver on next boot.
"""

import sys
import os
import time
import json
import math
import struct
import argparse
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smbus2 import SMBus
from src.config import (
    HMC5883L_I2C_BUS, HMC5883L_I2C_ADDR,
    MAGNETIC_DECLINATION_DEG, BASE_DIR,
)

# QMC5883P registers
REG_CHIP_ID     = 0x00
REG_DATA_X_LSB  = 0x01
REG_STATUS      = 0x09
REG_CTRL1       = 0x0A
REG_CTRL2       = 0x0B

SCALE = 3000.0  # LSB per Gauss for 8G range

CALIBRATION_FILE = str(BASE_DIR / "mag_calibration.json")


def init_sensor(bus: SMBus, addr: int):
    """Initialize QMC5883P for continuous reading."""
    chip_id = bus.read_byte_data(addr, REG_CHIP_ID)
    print(f"  Chip ID: 0x{chip_id:02X} (expected 0x80)")

    # Soft reset
    bus.write_byte_data(addr, REG_CTRL2, 0x80)
    time.sleep(0.05)

    # Range = 8G, Set/Reset enable
    bus.write_byte_data(addr, REG_CTRL2, 0x08)

    # OSR=512, ODR=50Hz, Mode=Continuous
    ctrl1 = (0b00 << 6) | (0b01 << 2) | 0b11
    bus.write_byte_data(addr, REG_CTRL1, ctrl1)
    time.sleep(0.1)
    print("  QMC5883P initialized (50Hz, 8G range)")


def read_raw(bus: SMBus, addr: int):
    """Read raw X, Y, Z in Gauss. Returns None if not ready."""
    status = bus.read_byte_data(addr, REG_STATUS)
    if not (status & 0x01):
        return None
    raw = bus.read_i2c_block_data(addr, REG_DATA_X_LSB, 6)
    x = struct.unpack("<h", bytes(raw[0:2]))[0] / SCALE
    y = struct.unpack("<h", bytes(raw[2:4]))[0] / SCALE
    z = struct.unpack("<h", bytes(raw[4:6]))[0] / SCALE
    return x, y, z


def compute_heading(x, y, declination_deg=MAGNETIC_DECLINATION_DEG):
    """Compute heading from calibrated X, Y."""
    heading_rad = math.atan2(y, x)
    heading_deg = math.degrees(heading_rad) + declination_deg
    return heading_deg % 360


def load_existing_calibration():
    """Load existing calibration if available."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        return data
    return None


def save_calibration(offsets: dict):
    """Save calibration offsets to JSON file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"\n  Calibration saved to: {CALIBRATION_FILE}")


def cmd_dump(duration: float = 10.0):
    """Dump raw magnetometer readings for debugging."""
    print(f"\n=== RAW MAGNETOMETER DUMP ({duration}s) ===")
    print("  This shows uncalibrated X, Y, Z values and computed heading.")
    print("  Use this to see what the sensor is reading.\n")

    bus = SMBus(HMC5883L_I2C_BUS)
    init_sensor(bus, HMC5883L_I2C_ADDR)

    # Load existing calibration for comparison
    cal = load_existing_calibration()
    if cal:
        print(f"  Existing calibration: offset_x={cal['offset_x']:.4f}, "
              f"offset_y={cal['offset_y']:.4f}, offset_z={cal['offset_z']:.4f}")
    else:
        print("  No existing calibration found.")
    print()

    print(f"  {'Time':>6s}  {'rawX':>8s}  {'rawY':>8s}  {'rawZ':>8s}  "
          f"{'hdg_raw':>8s}  {'hdg_cal':>8s}")
    print("  " + "-" * 58)

    t0 = time.monotonic()
    count = 0
    while (time.monotonic() - t0) < duration:
        raw = read_raw(bus, HMC5883L_I2C_ADDR)
        if raw:
            x, y, z = raw
            hdg_raw = compute_heading(x, y)

            if cal:
                cx = x - cal["offset_x"]
                cy = y - cal["offset_y"]
                hdg_cal = compute_heading(cx, cy)
            else:
                hdg_cal = float("nan")

            elapsed = time.monotonic() - t0
            print(f"  {elapsed:6.1f}  {x:8.4f}  {y:8.4f}  {z:8.4f}  "
                  f"{hdg_raw:7.1f}°  {hdg_cal:7.1f}°")
            count += 1

        time.sleep(0.067)  # ~15Hz

    bus.close()
    print(f"\n  {count} readings collected.")


def cmd_calibrate(duration: float = 30.0):
    """Run hard-iron calibration by collecting min/max over a rotation."""
    print(f"\n{'='*60}")
    print("  MAGNETOMETER HARD-IRON CALIBRATION")
    print(f"{'='*60}")
    print(f"\n  Duration: {duration} seconds")
    print("  Instructions:")
    print("    1. Keep the buoy LEVEL (don't tilt)")
    print("    2. SLOWLY rotate it 360° in the horizontal plane")
    print("    3. Try to complete at least 2 full rotations")
    print("    4. The more even/slow the rotation, the better\n")

    input("  Press ENTER when ready to start...")
    print()

    bus = SMBus(HMC5883L_I2C_BUS)
    init_sensor(bus, HMC5883L_I2C_ADDR)

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    t0 = time.monotonic()
    count = 0
    last_print = 0

    print("  Collecting data... rotate NOW!")
    print(f"  {'Time':>5s}  {'X range':>16s}  {'Y range':>16s}  {'Z range':>16s}")
    print("  " + "-" * 58)

    while (time.monotonic() - t0) < duration:
        raw = read_raw(bus, HMC5883L_I2C_ADDR)
        if raw:
            x, y, z = raw
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_z, max_z = min(min_z, z), max(max_z, z)
            count += 1

            elapsed = time.monotonic() - t0
            if elapsed - last_print >= 2.0:
                last_print = elapsed
                remaining = duration - elapsed
                print(f"  {elapsed:5.0f}s  "
                      f"X[{min_x:+.3f},{max_x:+.3f}]  "
                      f"Y[{min_y:+.3f},{max_y:+.3f}]  "
                      f"Z[{min_z:+.3f},{max_z:+.3f}]  "
                      f"({remaining:.0f}s left)")

        time.sleep(0.033)  # ~30Hz collection

    bus.close()

    if count < 10:
        print("\n  ERROR: Too few readings collected! Check sensor connection.")
        return

    # Compute hard-iron offsets (center of min/max ellipsoid)
    offset_x = (min_x + max_x) / 2
    offset_y = (min_y + max_y) / 2
    offset_z = (min_z + max_z) / 2

    # Compute range (useful for soft-iron check)
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    print(f"\n{'='*60}")
    print("  CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples collected: {count}")
    print(f"\n  Hard-iron offsets (to subtract from raw readings):")
    print(f"    offset_x = {offset_x:+.4f} Gauss")
    print(f"    offset_y = {offset_y:+.4f} Gauss")
    print(f"    offset_z = {offset_z:+.4f} Gauss")
    print(f"\n  Axis ranges (should be roughly equal if no soft-iron):")
    print(f"    X range = {range_x:.4f} Gauss")
    print(f"    Y range = {range_y:.4f} Gauss")
    print(f"    Z range = {range_z:.4f} Gauss")

    # Check quality
    if range_x < 0.05 or range_y < 0.05:
        print("\n  ⚠ WARNING: Very small X or Y range!")
        print("    Did you rotate the buoy enough? Try again with a full 360° rotation.")
    else:
        ratio = max(range_x, range_y) / min(range_x, range_y)
        if ratio > 2.0:
            print(f"\n  ⚠ WARNING: X/Y range ratio = {ratio:.1f} (ideally < 1.5)")
            print("    This suggests significant soft-iron distortion.")
            print("    Calibration may be less accurate.")
        else:
            print(f"\n  ✓ X/Y range ratio = {ratio:.2f} (good, < 2.0)")

    # Save
    cal_data = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "offset_z": offset_z,
        "range_x": range_x,
        "range_y": range_y,
        "range_z": range_z,
        "samples": count,
        "duration_s": duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "declination_deg": MAGNETIC_DECLINATION_DEG,
    }
    save_calibration(cal_data)

    # Quick test: show corrected heading
    print("\n  Quick test with new calibration:")
    bus = SMBus(HMC5883L_I2C_BUS)
    init_sensor(bus, HMC5883L_I2C_ADDR)
    time.sleep(0.2)

    for i in range(5):
        raw = read_raw(bus, HMC5883L_I2C_ADDR)
        if raw:
            x, y, z = raw
            hdg_raw = compute_heading(x, y)
            hdg_cal = compute_heading(x - offset_x, y - offset_y)
            print(f"    Raw heading: {hdg_raw:6.1f}°  →  Calibrated: {hdg_cal:6.1f}°")
        time.sleep(0.2)

    bus.close()
    print(f"\n  Done! Restart the ASV controller to use new calibration.")


def cmd_test():
    """Test current calibration by showing live heading."""
    print("\n=== TESTING CURRENT CALIBRATION ===")
    cal = load_existing_calibration()
    if cal:
        print(f"  Loaded calibration from: {CALIBRATION_FILE}")
        print(f"  Offsets: x={cal['offset_x']:+.4f}, y={cal['offset_y']:+.4f}, "
              f"z={cal['offset_z']:+.4f}")
        print(f"  Calibrated on: {cal.get('timestamp', 'unknown')}")
    else:
        print("  No calibration file found! Using zero offsets.")
        cal = {"offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0}

    print("\n  Showing live heading (Ctrl+C to stop)...\n")

    bus = SMBus(HMC5883L_I2C_BUS)
    init_sensor(bus, HMC5883L_I2C_ADDR)

    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

    try:
        while True:
            raw = read_raw(bus, HMC5883L_I2C_ADDR)
            if raw:
                x, y, z = raw
                cx = x - cal["offset_x"]
                cy = y - cal["offset_y"]
                hdg = compute_heading(cx, cy)

                # Cardinal direction
                idx = int((hdg + 11.25) / 22.5) % 16
                cardinal = directions[idx]

                # Simple bar for visualization
                bar_pos = int(hdg / 360 * 40)
                bar = "." * bar_pos + "█" + "." * (39 - bar_pos)

                print(f"\r  Heading: {hdg:6.1f}° {cardinal:>3s}  [{bar}]  "
                      f"raw=({x:+.3f},{y:+.3f},{z:+.3f})  ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n  Stopped.")
    finally:
        bus.close()


def main():
    parser = argparse.ArgumentParser(
        description="ASV Magnetometer Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dump", action="store_true",
                        help="Dump raw magnetometer readings")
    parser.add_argument("--test", action="store_true",
                        help="Test current calibration with live heading")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Calibration duration in seconds (default: 30)")
    parser.add_argument("--dump-duration", type=float, default=10.0,
                        help="Dump duration in seconds (default: 10)")

    args = parser.parse_args()

    if args.dump:
        cmd_dump(args.dump_duration)
    elif args.test:
        cmd_test()
    else:
        cmd_calibrate(args.duration)


if __name__ == "__main__":
    main()
