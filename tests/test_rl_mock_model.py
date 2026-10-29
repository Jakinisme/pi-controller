"""Pin the stand-in vessel model's body -> ENU mapping to the project convention.

``tools/mock_smoke.py::SimpleVessel`` is the only plant in the mock smoke run, so a
sign error in it silently flips what "+sway" means and any conclusion drawn from that
run is void. These tests feed the stand-in pure sway and pure surge and compare the
resulting ground velocity against ``src/rl/frames.py`` — the mapping the DRL env, the
obs builder and the heading/wrap conventions all agree on.

Project convention (frames.py module docstring, thrust.py ThrustCommand):
    sway + = starboard = right;  psi = 0 -> bow north;  psi = +90 deg -> bow east.

Pure Python, no hardware, no network, no ONNX.
"""

from __future__ import annotations

import math

import pytest

from src.control.thrust import ThrustCommand
from src.rl.frames import body_to_enu, enu_to_body
from tools.mock_smoke import SimpleVessel

HOME_LAT, HOME_LON = -6.8969, 112.0521
EARTH_R = 6_371_000.0


def _offset_m(vessel):
    """(east_m, north_m) travelled from the start, flat-earth like frames.latlon_to_enu."""
    north = math.radians(vessel.lat - HOME_LAT) * EARTH_R
    east = math.radians(vessel.lon - HOME_LON) * EARTH_R * math.cos(math.radians(HOME_LAT))
    return east, north


def _travelled(surge, sway, heading_deg, n=200, dt=0.05):
    v = SimpleVessel(HOME_LAT, HOME_LON, heading=heading_deg)
    cmd = ThrustCommand(surge=surge, sway=sway, yaw=0.0)
    for _ in range(n):
        v.step(cmd, dt)
    return _offset_m(v)


def _expected(surge, sway, heading_deg):
    """What the project mapping says after settling: velocity, per second."""
    v = SimpleVessel(HOME_LAT, HOME_LON, heading=heading_deg)
    v.u = surge * v.MAX_SPEED_MPS
    v.v = sway * v.MAX_SPEED_MPS * v.SWAY_GAIN
    return body_to_enu(v.u, v.v, math.radians(heading_deg))


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("axis", ["surge", "sway"])
def test_stand_in_ground_velocity_matches_frames(heading_deg, axis):
    """Settled velocity must match ``frames.body_to_enu`` in sign on both axes.

    Only the sign is asserted: the first-order lag reaches ~99.99% in 10 s, so the
    magnitude comparison would be a test of TAU_S, which is not the point here.
    """
    surge, sway = (1.0, 0.0) if axis == "surge" else (0.0, 1.0)
    east, north = _travelled(surge, sway, heading_deg)
    exp_e, exp_n = _expected(surge, sway, heading_deg)
    assert math.copysign(1.0, east) == math.copysign(1.0, float(exp_e)) or \
        abs(exp_e) < 1e-9 and abs(east) < 1e-3, (
            f"{axis} at heading {heading_deg}: east {east:+.3f} m, "
            f"frames.body_to_enu says {float(exp_e):+.6f} m/s")
    assert math.copysign(1.0, north) == math.copysign(1.0, float(exp_n)) or \
        abs(exp_n) < 1e-9 and abs(north) < 1e-3, (
            f"{axis} at heading {heading_deg}: north {north:+.3f} m, "
            f"frames.body_to_enu says {float(exp_n):+.6f} m/s")


def test_pure_sway_at_psi_zero_goes_starboard():
    """psi=0 (bow north): sway + must move the hull EAST (right of the bow)."""
    east, north = _travelled(0.0, 1.0, 0.0)
    assert east > 0.05, f"+sway moved east {east:+.3f} m: expected starboard (east)"
    assert abs(north) < 1e-3


def test_pure_sway_at_psi_90_goes_south():
    """psi=90 (bow east): starboard is south, so sway + must move the hull SOUTH."""
    east, north = _travelled(0.0, 1.0, 90.0)
    assert north < -0.05, f"+sway moved north {north:+.3f} m: expected starboard (south)"
    assert abs(east) < 1e-3


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, 180.0, 270.0])
def test_pure_surge_follows_the_bow(heading_deg):
    """Sanity anchor: surge + always along the bow, independent of the sway sign.

    Compared against ``_expected`` rather than an axis sign, so the cos(90 deg) ~ 1e-17
    zero-crossing of the projection does not decide the test.
    """
    east, north = _travelled(1.0, 0.0, heading_deg)
    exp_e, exp_n = _expected(1.0, 0.0, heading_deg)
    assert abs(east) == pytest.approx(abs(float(exp_e)) * 10.0, rel=0.1)
    assert abs(north) == pytest.approx(abs(float(exp_n)) * 10.0, rel=0.1)


def test_frames_body_to_enu_is_the_inverse_of_enu_to_body():
    """Guard on the convention itself: round-trip must be exact to float noise."""
    e, n = body_to_enu(0.7, -0.3, math.radians(37.0))
    u, v = enu_to_body(e, n, math.radians(37.0))
    assert float(u) == pytest.approx(0.7, abs=1e-12)
    assert float(v) == pytest.approx(-0.3, abs=1e-12)
