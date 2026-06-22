"""
Waypoint Navigation State Machine + Dynamic Positioning.

Modes:
  IDLE           - No active mission, thrusters at zero
  WAYPOINT_NAV   - Following a list of waypoints
  STATION_KEEP   - Dynamic positioning (hold position + heading)
  MANUAL         - Direct thrust commands from dashboard (bypass PID)
"""

import math
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.utils.geo import haversine_distance, bearing, heading_error
from src.config import (
    WAYPOINT_ARRIVAL_RADIUS_M,
    WAYPOINT_MAX_SPEED,
    WAYPOINT_SLOWDOWN_RADIUS_M,
    STATION_KEEPING_DEADBAND_M,
)
from src.utils.logger import setup_logger

log = setup_logger("nav")


class NavigationState(Enum):
    IDLE = auto()
    WAYPOINT_NAV = auto()
    STATION_KEEP = auto()
    MANUAL = auto()


@dataclass
class Waypoint:
    """A navigation waypoint."""
    latitude: float
    longitude: float
    name: str = ""
    heading_hold: Optional[float] = None  # Desired heading at waypoint (degrees), None = auto


@dataclass
class NavigationOutput:
    """Output from the navigation module to the thrust computer."""
    desired_speed: float = 0.0       # m/s (for waypoint mode)
    desired_heading: float = 0.0     # degrees (for waypoint mode)
    desired_north_m: float = 0.0     # meters north error (for station-keeping)
    desired_east_m: float = 0.0      # meters east error (for station-keeping)
    hold_heading: float = 0.0        # degrees (for station-keeping)
    mode: NavigationState = NavigationState.IDLE


class WaypointNavigator:
    """Navigation state machine for waypoint following and dynamic positioning."""

    def __init__(
        self,
        arrival_radius: float = WAYPOINT_ARRIVAL_RADIUS_M,
        max_speed: float = WAYPOINT_MAX_SPEED,
        slowdown_radius: float = WAYPOINT_SLOWDOWN_RADIUS_M,
        dp_deadband: float = STATION_KEEPING_DEADBAND_M,
    ):
        self._arrival_radius = arrival_radius
        self._max_speed = max_speed
        self._slowdown_radius = slowdown_radius
        self._dp_deadband = dp_deadband

        self.state = NavigationState.IDLE
        self.waypoints: List[Waypoint] = []
        self.current_wp_index: int = 0
        self.output = NavigationOutput()

        # Station-keeping target
        self._hold_lat: float = 0.0
        self._hold_lon: float = 0.0
        self._hold_heading: float = 0.0

        # Last known position (for distance_to_waypoint)
        self._last_lat: float = 0.0
        self._last_lon: float = 0.0

    def start_waypoint_nav(self, waypoints: List[Waypoint]):
        """Begin waypoint navigation with a new waypoint list."""
        if not waypoints:
            log.warning("Cannot start nav: empty waypoint list")
            return
        self.waypoints = waypoints
        self.current_wp_index = 0
        self.state = NavigationState.WAYPOINT_NAV
        log.info("Waypoint nav started: %d waypoints", len(waypoints))

    def start_station_keeping(self, lat: float, lon: float, heading: float):
        """Begin dynamic positioning at the given coordinates."""
        self._hold_lat = lat
        self._hold_lon = lon
        self._hold_heading = heading
        self.state = NavigationState.STATION_KEEP
        log.info("Station keeping at (%.6f, %.6f) heading %.1f", lat, lon, heading)

    def set_manual(self):
        """Switch to manual mode (PID bypassed, direct thrust from dashboard)."""
        self.state = NavigationState.MANUAL
        log.info("Manual mode activated")

    def stop(self):
        """Stop all navigation, go to IDLE."""
        self.state = NavigationState.IDLE
        self.output = NavigationOutput()
        log.info("Navigation stopped -> IDLE")

    def update(self, lat: float, lon: float, heading: float) -> NavigationOutput:
        """Update navigation state based on current position.
        
        Args:
            lat: Current latitude (degrees).
            lon: Current longitude (degrees).
            heading: Current fused heading (degrees).
        
        Returns:
            NavigationOutput with desired speed/heading or position errors.
        """
        self._last_lat = lat
        self._last_lon = lon

        if lat == 0.0 and lon == 0.0:
            # No GPS fix, stay idle
            return self.output

        if self.state == NavigationState.WAYPOINT_NAV:
            self._update_waypoint_nav(lat, lon, heading)

        elif self.state == NavigationState.STATION_KEEP:
            self._update_station_keeping(lat, lon, heading)

        elif self.state == NavigationState.IDLE:
            self.output = NavigationOutput(mode=NavigationState.IDLE)

        return self.output

    def _update_waypoint_nav(self, lat: float, lon: float, heading: float):
        """Waypoint following logic."""
        if self.current_wp_index >= len(self.waypoints):
            log.info("All waypoints reached!")
            self.stop()
            return

        wp = self.waypoints[self.current_wp_index]
        dist = haversine_distance(lat, lon, wp.latitude, wp.longitude)
        brg = bearing(lat, lon, wp.latitude, wp.longitude)

        # Check arrival
        if dist < self._arrival_radius:
            log.info(
                "Arrived at waypoint %d/%d (%.6f, %.6f) dist=%.1fm",
                self.current_wp_index + 1, len(self.waypoints),
                wp.latitude, wp.longitude, dist,
            )
            self.current_wp_index += 1
            if self.current_wp_index >= len(self.waypoints):
                self.stop()
                return
            # Update to next waypoint
            wp = self.waypoints[self.current_wp_index]
            dist = haversine_distance(lat, lon, wp.latitude, wp.longitude)
            brg = bearing(lat, lon, wp.latitude, wp.longitude)

        # Speed scaling: slow down near waypoint
        if dist < self._slowdown_radius:
            speed_ratio = max(0.2, dist / self._slowdown_radius)
            speed = self._max_speed * speed_ratio
        else:
            speed = self._max_speed

        # Desired heading
        desired_heading = wp.heading_hold if wp.heading_hold is not None else brg

        self.output = NavigationOutput(
            desired_speed=speed,
            desired_heading=desired_heading,
            mode=NavigationState.WAYPOINT_NAV,
        )

    def _update_station_keeping(self, lat: float, lon: float, heading: float):
        """Dynamic positioning: compute NED position error."""
        # Convert GPS error to meters (north, east)
        dlat = lat - self._hold_lat
        dlon = lon - self._hold_lon
        north_m = dlat * (math.pi / 180) * 6_371_000
        east_m = dlon * (math.pi / 180) * 6_371_000 * math.cos(math.radians(self._hold_lat))

        # Apply deadband
        if abs(north_m) < self._dp_deadband:
            north_m = 0.0
        if abs(east_m) < self._dp_deadband:
            east_m = 0.0

        self.output = NavigationOutput(
            desired_north_m=north_m,
            desired_east_m=east_m,
            hold_heading=self._hold_heading,
            mode=NavigationState.STATION_KEEP,
        )

    def skip_waypoint(self):
        """Skip the current waypoint and move to the next."""
        if self.state == NavigationState.WAYPOINT_NAV and self.waypoints:
            self.current_wp_index += 1
            log.info("Skipped to waypoint %d", self.current_wp_index + 1)

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if (self.state == NavigationState.WAYPOINT_NAV
                and 0 <= self.current_wp_index < len(self.waypoints)):
            return self.waypoints[self.current_wp_index]
        return None

    @property
    def distance_to_waypoint(self) -> float:
        """Distance to current waypoint (requires last known position)."""
        wp = self.current_waypoint
        if wp and hasattr(self, '_last_lat'):
            return haversine_distance(self._last_lat, self._last_lon, wp.latitude, wp.longitude)
        return 0.0