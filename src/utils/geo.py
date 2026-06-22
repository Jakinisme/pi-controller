"""
Geospatial utility functions for ASV navigation.
All angles in degrees unless otherwise noted; distances in meters.
"""

import math
from typing import Tuple


EARTH_RADIUS_M = 6_371_000  # Mean Earth radius in meters


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two GPS points.
    
    Args:
        lat1, lon1: Start point (degrees).
        lat2, lon2: End point (degrees).
    
    Returns:
        Distance in meters.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing (forward azimuth) from point 1 to point 2.
    
    Args:
        lat1, lon1: Start point (degrees).
        lat2, lon2: End point (degrees).
    
    Returns:
        Bearing in degrees [0, 360).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)

    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def cross_track_error(
    lat_current: float,
    lon_current: float,
    lat_from: float,
    lon_from: float,
    lat_to: float,
    lon_to: float,
) -> float:
    """Calculate cross-track error (perpendicular distance from the path).
    
    Positive = current position is to the RIGHT of the path (from->to).
    Negative = current position is to the LEFT.
    
    Args:
        lat_current, lon_current: Current position (degrees).
        lat_from, lon_from: Path start point (degrees).
        lat_to, lon_to: Path end point (degrees).
    
    Returns:
        Cross-track distance in meters.
    """
    d13 = haversine_distance(lat_from, lon_from, lat_current, lon_current) / EARTH_RADIUS_M
    theta13 = math.radians(bearing(lat_from, lon_from, lat_current, lon_current))
    theta12 = math.radians(bearing(lat_from, lon_from, lat_to, lon_to))

    xte = math.asin(math.sin(d13) * math.sin(theta13 - theta12)) * EARTH_RADIUS_M
    return xte


def lat_lon_offset(lat: float, lon: float, north_m: float, east_m: float) -> Tuple[float, float]:
    """Offset a GPS position by meters in the local NED frame.
    
    Args:
        lat, lon: Reference point (degrees).
        north_m: Offset in meters north.
        east_m: Offset in meters east.
    
    Returns:
        (new_lat, new_lon) in degrees.
    """
    new_lat = lat + (north_m / EARTH_RADIUS_M) * (180 / math.pi)
    new_lon = lon + (east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180 / math.pi)
    return new_lat, new_lon


def heading_error(current_heading: float, desired_heading: float) -> float:
    """Calculate shortest heading error with wrap-around.
    
    Args:
        current_heading: Current heading in degrees [0, 360).
        desired_heading: Desired heading in degrees [0, 360).
    
    Returns:
        Heading error in degrees [-180, 180]. Positive = turn right.
    """
    error = (desired_heading - current_heading + 180) % 360 - 180
    return error
