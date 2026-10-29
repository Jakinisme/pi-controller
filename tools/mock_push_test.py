"""Mock push test: scripted goal 20 m dead ahead, real ``ControllerMux``, two stand-ins.

Same scripted mission, two vessel stand-ins:

  A  ``SimpleVessel``      the model in ``tools/mock_smoke.py`` (imported, NOT edited):
                           +sway -> negative lateral speed (port). 
  B  ``CorrectedVessel``   identical except sway obeys the project convention
                           (``+sway = starboard``, ``src/rl/frames.py::body_to_enu``).

Both are holistic X-config stand-ins (4 thrusters, surge/sway/yaw combined), i.e. the
same nonholonomic-lite plant; only the sway sign differs, so any difference in the run
is attributable to that sign alone. Because ``SimpleVessel`` is imported verbatim, a
later fix to ``mock_smoke.py`` flows into this tool automatically.

    .venv/Scripts/python.exe tools/mock_push_test.py
    .venv/Scripts/python.exe tools/mock_push_test.py --goal-n 20 --seconds 20 --print-limit 10

No hardware, no network, no real vessel. Numbers describe this laptop and a stand-in
model.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# Hardware mock FIRST, before anything from src/ is imported.
# ---------------------------------------------------------------------------
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from hardware_mock import install  # noqa: E402

install()

from src import config as cfg  # noqa: E402
from src.rl.frames import body_to_enu  # noqa: E402
from tools.mock_smoke import (  # noqa: E402
    EARTH_R, HOME_LAT, HOME_LON, SimpleVessel, _make_waypoints, _pct,
)


class CorrectedVessel(SimpleVessel):
    """``SimpleVessel`` with the lateral term made sign-correct.

    Everything else (first-order lag, ``SWAY_GAIN``, yaw mixing, projection) is
    inherited unchanged. The parent's ``step`` is not overridden; only the sway
    sign lives elsewhere, so the whole body->ENU block is restated here and built
    from ``src/rl/frames.py::body_to_enu`` — the project's own mapping.
    """

    def step(self, cmd, dt):
        u_t = cmd.surge * self.MAX_SPEED_MPS
        v_t = cmd.sway * self.MAX_SPEED_MPS * self.SWAY_GAIN     # +sway = +starboard
        a = dt / (self.TAU_S + dt)
        self.u += (u_t - self.u) * a
        self.v += (v_t - self.v) * a
        self.yaw_rate = cmd.yaw * self.YAW_RATE_DEG_S
        self.heading = (self.heading + self.yaw_rate * dt) % 360.0

        ve, vn = body_to_enu(self.u, self.v, math.radians(self.heading))
        ve, vn = float(ve), float(vn)
        self.lat += vn * dt / EARTH_R * (180.0 / math.pi)
        self.lon += ve * dt / (EARTH_R * math.cos(math.radians(self.lat))) * (180.0 / math.pi)
        return math.hypot(self.u, self.v)


def _bearing_m(lat, lon, lat0, lon0):
    """(east_m, north_m) offset, flat-earth like ``frames.latlon_to_enu``."""
    north = math.radians(lat - lat0) * EARTH_R
    east = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return east, north


def _scripted_run(sysm, controller, vessel_cls, *, goal_n, seconds, start_heading,
                  lpad=4):
    """Fly one scripted leg: hold heading, then steer to (goal_n ahead, 0 cross)."""
    sysm.navigator.stop()
    sysm.navigator.start_waypoint_nav(_make_waypoints())
    sysm.controller.set_controller_request(controller)
    vessel = vessel_cls(HOME_LAT, HOME_LON, heading=start_heading)
    sysm.gps.data.true_course = None  # mux falls back to heading, like the smoke

    interval = 1.0 / cfg.FUSION_UPDATE_RATE_HZ
    # `now` must be on the adapter's clock (time.monotonic): _build_est puts it in
    # `t` and RLController._check compares it to time.monotonic() -> watchdog otherwise.
    t0, rows = time.monotonic(), []
    for i in range(int(round(seconds * cfg.FUSION_UPDATE_RATE_HZ))):
        now = t0 + i * interval
        t = i * interval
        state, course = vessel.to_state(t)
        nav = sysm.navigator.update(lat=state.latitude, lon=state.longitude,
                                    heading=state.heading, gps_valid=True)
        cmd = sysm.controller.compute(nav, state, now=now)
        e, n = _bearing_m(state.latitude, state.longitude, HOME_LAT, HOME_LON)
        if i % cfg.FUSION_UPDATE_RATE_HZ == 0:
            rows.append({
                "t": t, "range": math.hypot(goal_n - n, 0.0 - e),
                "east": e, "north": n, "hdg": state.heading,
                "surge": cmd.surge, "sway": cmd.sway, "yaw": cmd.yaw,
                "v": vessel.v, "src": sysm.controller.source,
                "fault": sysm.controller.fault or "-",
                "nav": sysm.navigator.state.name,
            })
        vessel.step(cmd, interval)
        t += interval
    return rows, sysm.navigator.state.name


def _emit(tag, rows, limit):
    print(f"-- {tag} --")
    print("   t     range  north   east   hdg   surge    sway     yaw     v    src/fault")
    step = max(1, len(rows) // max(1, limit))
    for r in rows[::step][:limit]:
        print(f"  {r['t']:4.1f}s  {r['range']:6.2f} {r['north']:7.2f} {r['east']:6.2f} "
              f"{r['hdg']:6.1f} {r['surge']:+7.3f} {r['sway']:+7.3f} {r['yaw']:+7.3f} "
              f"{r['v']:+6.3f}  {r['src']}/{r['fault']}")
    r = rows[-1]
    print(f"  final {r['t']:.1f}s: range={r['range']:.2f} m  north={r['north']:+.2f}  "
          f"east={r['east']:+.2f}  |head_err|={abs(((r['hdg'] + 180) % 360) - 180):.1f} deg  "
          f"final nav={r['nav']}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="Mock push test: SimpleVessel vs corrected sway sign")
    ap.add_argument("--goal-n", type=float, default=20.0, help="goal north offset [m]")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--heading", type=float, default=0.0, help="start heading [deg]")
    ap.add_argument("--print-limit", type=int, default=21, help="rows printed per run")
    ap.add_argument("--controller", default="ppo", choices=("ppo", "pid"))
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or os.path.join(REPO, "logs", "rl_push")

    cfg.RL_ENABLED = True
    cfg.RL_DEFAULT_CONTROLLER = "ppo"
    cfg.RL_GEOFENCE_M = 1000.0   # isolate the vessel model from the 20 m mux fence
    cfg.RL_LOG_DIR = log_dir

    from src.comms.firebase_bridge import FirebaseBridge
    from src.main import ASVSystem

    FirebaseBridge(enabled=False)
    sysm = ASVSystem()
    if not getattr(sysm.rl_adapter, "available", False):
        print(f"FATAL: RL adapter unavailable ({sysm.rl_adapter.reason})", file=sys.stderr)
        return 2

    class _GpsData:
        true_course = 0.0
        num_satellites = 12
        altitude = 0.0
        valid = True

    class _Gps:
        data = _GpsData()

    sysm.controller.gps = _Gps()
    sysm.gps = _Gps()

    def _run(vessel_cls):
        return _scripted_run(sysm, args.controller, vessel_cls,
                             goal_n=args.goal_n, seconds=args.seconds,
                             start_heading=args.heading)

    a_rows, a_nav = _run(SimpleVessel)
    b_rows, b_nav = _run(CorrectedVessel)

    pol_ms = [float(getattr(sysm.rl_adapter, "last_inference_ms", 0.0) or 0.0)]
    print("=" * 78)
    print("MOCK PUSH TEST  (MOCK: no vessel, no hardware, no GPS, no network)")
    print("=" * 78)
    print(f"script : heading {args.heading:.0f} deg -> goal {args.goal_n:.0f} m dead ahead, "
          f"{args.seconds:.0f} s at {cfg.FUSION_UPDATE_RATE_HZ} Hz, controller={args.controller}")
    print(f"models : A=SimpleVessel (sway + -> port), B=CorrectedVessel (sway + -> starboard)")
    ra = _emit("A  SimpleVessel  (as shipped in tools/mock_smoke.py)", a_rows, args.print_limit)
    print()
    rb = _emit("B  CorrectedVessel  (sway sign per src/rl/frames.body_to_enu)", b_rows, args.print_limit)
    print()
    print(f"final range : A {ra['range']:.2f} m   B {rb['range']:.2f} m   "
          f"(B-A {rb['range'] - ra['range']:+.2f} m)")
    print(f"final cross : A east {ra['east']:+.2f} m   B east {rb['east']:+.2f} m")
    print(f"peak sway   : A {max(abs(r['sway']) for r in a_rows):.3f}   "
          f"B {max(abs(r['sway']) for r in b_rows):.3f}")
    print(f"nav state   : A {a_nav}   B {b_nav}")
    print(f"inference_ms (last): {pol_ms[0]:.3f}")
    print("-" * 78)
    print("MOCK: stand-in physics on this laptop only. No claim about the real vessel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
