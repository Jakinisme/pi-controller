"""Mock-hardware smoke run for the RL (PPO) control path.

Drives the REAL ``ASVSystem`` — ``ControllerMux.compute()``, the vendored
``RLController``, the navigator, the PID path — at the configured control rate
with hardware mocked out and Firebase disabled. A scripted square mission
(4 waypoints, ~50 m legs around the mock home) is flown against a simple
first-order synthetic vessel model: the state is advanced from the *applied*
thrust command, so the waypoints actually get reached and the mux sees a
progressing, changing state.

    .venv/Scripts/python.exe tools/mock_smoke.py --seconds 60

Prints duration, tick count, the source and fault histograms, inference
p50/p95/max, control-loop period p50/p95/max, waypoints reached and the path of
the CSV it wrote. Exits non-zero if any emitted command is non-finite or outside
[-1, 1].

WHAT THIS IS NOT: there is no vessel, no hardware, no GPS and no network. Every
number below comes from a stand-in physics model written here, not from the
simulator the policy trained in and not from the real boat. The timing numbers
are this laptop's, not the Pi 3B's. Nothing here is evidence about real-vessel
behaviour; it only shows the wiring runs end to end and the safety layer does
not emit an illegal command.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Hardware mock FIRST, before anything from src/ is imported (see test.py).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hardware_mock import install  # noqa: E402

install()

import numpy as np  # noqa: E402

from src import config as cfg  # noqa: E402
from src.control.navigation import NavigationState, Waypoint  # noqa: E402
from src.sensors.fusion import FusedState  # noqa: E402

HOME_LAT = -6.8969
HOME_LON = 112.0521
EARTH_R = 6_371_000.0
# ~50 m square around home: N, E, S, back to W of home.
LEG_M = 50.0


def _make_waypoints():
    """4-waypoint square, ~50 m legs, returning near the mock home."""
    dlat = LEG_M / EARTH_R * (180.0 / math.pi)
    dlon = LEG_M / (EARTH_R * math.cos(math.radians(HOME_LAT))) * (180.0 / math.pi)
    return [
        Waypoint(HOME_LAT + dlat, HOME_LON, name="N", heading_hold=None),
        Waypoint(HOME_LAT + dlat, HOME_LON + dlon, name="NE", heading_hold=None),
        Waypoint(HOME_LAT, HOME_LON + dlon, name="SE", heading_hold=None),
        Waypoint(HOME_LAT, HOME_LON, name="S", heading_hold=None),
    ]


class SimpleVessel:
    """First-order speed model: applied command -> target velocity -> position.

    Left/right and fore/aft are in the body frame using the same convention the
    command vector uses (surge + = forward along bow, sway + = starboard),
    heading in degrees clockwise from north. Deliberately crude: this exists so
    waypoints advance, not to be physics.
    """

    MAX_SPEED_MPS = 1.5
    TAU_S = 1.0          # first-order time constant
    YAW_RATE_DEG_S = 20.0
    SWAY_GAIN = 0.2      # a hull barely slides sideways: lateral response << forward

    def __init__(self, lat, lon, heading=0.0):
        self.lat = lat
        self.lon = lon
        self.heading = heading
        self.u = 0.0     # forward, m/s
        self.v = 0.0     # starboard, m/s
        self.yaw_rate = 0.0

    def step(self, cmd, dt):
        u_t = cmd.surge * self.MAX_SPEED_MPS
        # sway + = starboard (+) per thrust.py / frames.body_to_enu, so v_t keeps the
        # sign: the earlier negation here inverted the lateral axis.
        v_t = cmd.sway * self.MAX_SPEED_MPS * self.SWAY_GAIN
        a = dt / (self.TAU_S + dt)
        self.u += (u_t - self.u) * a
        self.v += (v_t - self.v) * a
        self.yaw_rate = cmd.yaw * self.YAW_RATE_DEG_S
        self.heading = (self.heading + self.yaw_rate * dt) % 360.0

        # body -> ENU (x east, y north); heading CW from north.
        psi = math.radians(self.heading)
        vn = self.u * math.cos(psi) - self.v * math.sin(psi)
        ve = self.u * math.sin(psi) + self.v * math.cos(psi)
        self.lat += vn * dt / EARTH_R * (180.0 / math.pi)
        self.lon += ve * dt / (EARTH_R * math.cos(math.radians(self.lat))) * (180.0 / math.pi)
        return math.hypot(self.u, self.v)

    def to_state(self, t_mono):
        speed = math.hypot(self.u, self.v)
        course = (math.degrees(math.atan2(math.sin(math.radians(self.heading)),
                                          math.cos(math.radians(self.heading))))) % 360.0
        return FusedState(
            latitude=self.lat, longitude=self.lon, heading=self.heading,
            speed=speed, yaw_rate=self.yaw_rate, timestamp=time.time(),
            gps_valid=True, imu_valid=True, mag_valid=True,
        ), course


def _pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser(description="Mock-hardware RL smoke run")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="wall-clock seconds to run (default 60)")
    ap.add_argument("--log-dir", default=None,
                    help="CSV output dir (default: <repo>/logs/rl_smoke)")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = args.log_dir or os.path.join(repo, "logs", "rl_smoke")

    # Force the RL path on, for this process only. Deployment default stays
    # RL_ENABLED=False / controller "pid" in src/config.py.
    cfg.RL_ENABLED = True
    cfg.RL_DEFAULT_CONTROLLER = "ppo"
    cfg.RL_LOG_DIR = log_dir

    from src.comms.firebase_bridge import FirebaseBridge
    from src.main import ASVSystem

    # No credentials, no network: the bridge stays disabled.
    FirebaseBridge(enabled=False)

    sysm = ASVSystem()
    sysm.rl_logger = __import__("src.rl.logger", fromlist=["RLLogger"]).RLLogger(log_dir)
    sysm.controller.rl_logger = sysm.rl_logger
    sysm.gps.true_course = None  # not used; mux falls back to heading

    # A stand-in GPS so the mux's COG lookup does not explode on the mock object.
    class _GpsData:  # noqa: D401
        true_course = 0.0
        num_satellites = 12
        altitude = 0.0
        valid = True

    class _Gps:
        data = _GpsData()

    sysm.controller.gps = _Gps()
    sysm.gps = _Gps()

    available = getattr(sysm.rl_adapter, "available", False)
    reason = getattr(sysm.rl_adapter, "reason", None)
    if not available:
        print(f"FATAL: RL adapter unavailable ({reason}); cannot run the smoke.",
              file=sys.stderr)
        return 2

    interval = 1.0 / cfg.FUSION_UPDATE_RATE_HZ
    n_ticks = int(round(args.seconds * cfg.FUSION_UPDATE_RATE_HZ))

    def _run(controller):
        """Fly the square mission with one controller. Returns a metrics dict."""
        sysm.navigator.start_waypoint_nav(_make_waypoints())
        sysm.controller.set_controller_request(controller)
        vessel = SimpleVessel(HOME_LAT, HOME_LON, heading=0.0)

        sources, faults, infer_ms, periods = Counter(), Counter(), [], []
        wp_reached, bad = 0, 0
        last_wp = sysm.navigator.current_wp_index

        t_start = time.monotonic()
        next_tick, prev_tick = t_start, None
        for _ in range(n_ticks):
            # real-time pacing, like the asyncio loop's sleep
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            tick_t = time.monotonic()
            if prev_tick is not None:
                periods.append(tick_t - prev_tick)
            prev_tick = tick_t

            state, course = vessel.to_state(tick_t)
            sysm.gps.data.true_course = course

            nav_output = sysm.navigator.update(
                lat=state.latitude, lon=state.longitude,
                heading=state.heading, gps_valid=state.gps_valid,
            )
            cmd = sysm.controller.compute(nav_output, state, now=tick_t)

            for a in (cmd.surge, cmd.sway, cmd.yaw):
                if not math.isfinite(a) or not (-1.0 <= a <= 1.0):
                    bad += 1

            sources[sysm.controller.source] += 1
            faults[sysm.controller.fault or ""] += 1
            if sysm.controller.source == "policy":
                # last_inference_ms is the wall time of the last adapter.step() —
                # the same value the mux's slow_inference branch reads.
                infer_ms.append(float(getattr(sysm.rl_adapter, "last_inference_ms", 0.0) or 0.0))

            cur_wp = sysm.navigator.current_wp_index
            wp_reached += max(0, cur_wp - last_wp)
            last_wp = cur_wp

            vessel.step(cmd, interval)
            next_tick += interval
            if controller == "ppo" and sysm.navigator.state.name == "IDLE" and wp_reached == 0 \
                    and sources.get("zero"):
                break  # geofence latch: the run is over

        return {
            "duration": time.monotonic() - t_start,
            "sources": sources, "faults": faults, "infer_ms": infer_ms,
            "periods": periods, "wp_reached": wp_reached, "bad": bad,
            "nav_state": sysm.navigator.state.name, "ticks": sum(sources.values()),
        }

    pol = _run("ppo")
    sysm.rl_logger.flush()
    csv_path = getattr(sysm.rl_logger, "path", None)
    sysm.rl_logger.close()
    pid = _run("pid")

    def _emit(tag, m):
        print(f"-- {tag} --")
        print(f"  duration / ticks : {m['duration']:.2f} s / {m['ticks']}")
        print(f"  sources          : {dict(m['sources'])}")
        print(f"  faults           : {dict(m['faults'])}")
        print(f"  waypoints reached: {m['wp_reached']}/4   final nav state: {m['nav_state']}")
        if m["infer_ms"]:
            print(f"  inference_ms     : p50={_pct(m['infer_ms'],50):.3f} "
                  f"p95={_pct(m['infer_ms'],95):.3f} max={max(m['infer_ms']):.3f} "
                  f"(n={len(m['infer_ms'])})")
        if m["periods"]:
            print(f"  loop period p50/p95/max: {_pct(m['periods'],50)*1000:.2f}/"
                  f"{_pct(m['periods'],95)*1000:.2f}/{max(m['periods'])*1000:.2f} ms "
                  f"(target {interval*1000:.0f} ms)")

    print("=" * 68)
    print("RL MOCK-HARDWARE SMOKE  (MOCK: no vessel, no hardware, no GPS, no net)")
    print("=" * 68)
    print(f"tick rate : {cfg.FUSION_UPDATE_RATE_HZ} Hz  "
          f"(policy dt {sysm.rl_adapter.policy_dt_s} s)  "
          f"adapter_available={available}")
    _emit("ppo controller", pol)
    _emit("pid controller (same mission, control run)", pid)
    print(f"csv : {csv_path}")
    print("-" * 68)
    print("MOCK: no vessel, no hardware, no network. Numbers describe this laptop")
    print("and a stand-in physics model, NOT the real ASV.")
    print("-" * 68)
    if pol["bad"] or pid["bad"]:
        print(f"FAIL: {pol['bad'] + pid['bad']} command component(s) "
              f"non-finite or outside [-1,1]")
        return 1
    print("PASS: every emitted command finite and within [-1,1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
