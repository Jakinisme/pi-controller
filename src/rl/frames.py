"""Frame and heading conversions shared by sim, env, training and the Pi runtime.

Conventions (see drl/INSPECTION.md for the evidence, drl/CONTRACT.md for the
rule every later module codes against):

World frame: local ENU tangent plane, origin = first valid GPS fix.
    x = east  [m]
    y = north [m]

Heading psi: radians, clockwise from +y (north).
    psi = 0        -> bow points north
    psi = +pi/2    -> bow points east
This is the repo convention (compass degrees 0=N, clockwise positive, as used
by pi-control/src/utils/geo.py bearing() and heading_error()) expressed in
radians, so psi_rad = deg2rad(compass_heading_deg).

Body frame: x_b = bow (forward), y_b = starboard (right).
Body velocity: u = surge forward [m/s], v = sway starboard [m/s],
r = yaw rate = dpsi/dt [rad/s], clockwise positive.

Projection: equirectangular (flat Earth) with R = 6371000 m and cos(lat0)
scaling, identical to pi-control/src/utils/geo.py lat_lon_offset() and
navigation._update_station_keeping(). Valid on the scale of a survey site
(a few km); TODO re-check if the operating area exceeds ~10 km.

All functions accept scalars or numpy arrays and broadcast.
"""

import numpy as np

__all__ = [
    "EARTH_RADIUS_M",
    "wrap_deg",
    "wrap_pi",
    "heading_error_deg",
    "heading_error_rad",
    "heading_deg_to_psi",
    "psi_to_heading_deg",
    "latlon_to_enu",
    "enu_to_latlon",
    "enu_to_body",
    "body_to_enu",
    "goal_in_body",
]

# Mean Earth radius, matches pi-control/src/utils/geo.py:10
EARTH_RADIUS_M = 6_371_000.0


def wrap_deg(a):
    """Wrap degrees to [0, 360)."""
    return np.mod(np.asarray(a, dtype=float), 360.0)


def wrap_pi(a):
    """Wrap radians to [-pi, pi)."""
    return np.mod(np.asarray(a, dtype=float) + np.pi, 2.0 * np.pi) - np.pi


def heading_error_deg(current_deg, desired_deg):
    """Compass heading error in degrees, positive = turn starboard (right).

    Same formula as pi-control/src/utils/geo.py:108 heading_error().
    """
    return np.mod(np.asarray(desired_deg, float) - np.asarray(current_deg, float) + 180.0, 360.0) - 180.0


def heading_error_rad(psi, psi_des):
    """Heading error in radians, positive = turn starboard (right)."""
    return wrap_pi(np.asarray(psi_des, float) - np.asarray(psi, float))


def heading_deg_to_psi(heading_deg):
    """Compass heading in degrees [0,360) -> psi in radians [-pi,pi)."""
    return wrap_pi(np.deg2rad(np.asarray(heading_deg, dtype=float)))


def psi_to_heading_deg(psi):
    """psi in radians -> compass heading in degrees [0,360)."""
    return wrap_deg(np.rad2deg(np.asarray(psi, dtype=float)))


def latlon_to_enu(lat, lon, lat0, lon0):
    """WGS84 lat/lon [deg] -> local ENU (east, north) [m] about (lat0, lon0).

    Returns (east, north), each with the broadcast shape of the inputs.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    north = np.deg2rad(lat - lat0) * EARTH_RADIUS_M
    east = np.deg2rad(lon - lon0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
    return east, north


def enu_to_latlon(east, north, lat0, lon0):
    """Local ENU (east, north) [m] -> lat/lon [deg]. Inverse of latlon_to_enu."""
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    lat = lat0 + np.rad2deg(north / EARTH_RADIUS_M)
    lon = lon0 + np.rad2deg(east / (EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))
    return lat, lon


def enu_to_body(east, north, psi):
    """Rotate an ENU vector into the body frame -> (u, v).

    u is the component along the bow, v the component along starboard.
    """
    s = np.sin(psi)
    c = np.cos(psi)
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    u = east * s + north * c
    v = east * c - north * s
    return u, v


def body_to_enu(u, v, psi):
    """Rotate a body-frame vector (u, v) into ENU -> (east, north)."""
    s = np.sin(psi)
    c = np.cos(psi)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    east = u * s + v * c
    north = u * c - v * s
    return east, north


def goal_in_body(e_goal, n_goal, e, n, psi):
    """Goal offset (ENU positions) seen from the body frame -> (dx, dy).

    dx = along the bow [m], dy = toward starboard [m].
    """
    return enu_to_body(np.subtract(e_goal, e), np.subtract(n_goal, n), psi)
