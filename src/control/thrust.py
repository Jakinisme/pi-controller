"""Thrust Vector Computer.

Converts navigation outputs (desired speed/heading or position errors)
into high-level thrust commands: [surge, sway, yaw].

These are sent to ESP32 which handles motor mixing.
Thruster layout: + configuration rotated 45° CW (radial outward thrust).

Mixing:
# ✅ CORRECT mixing for push-pull catamaran
T_NW =  surge - sway + yaw    # front-left
T_NE =  surge + sway - yaw    # front-right
T_SW = -surge + sway - yaw    # rear-left  (surge negated!)
T_SE = -surge - sway + yaw    # rear-right (surge negated!)
"""

import math
import time
from dataclasses import dataclass

from src.control.pid import PIDController
from src.control.navigation import NavigationState, NavigationOutput
from src.sensors.fusion import FusedState
from src.config import (
    POSITION_PID_KP, POSITION_PID_KI, POSITION_PID_KD,
    POSITION_PID_INTEGRAL_LIMIT, POSITION_PID_OUTPUT_LIMIT,
    HEADING_PID_KP, HEADING_PID_KI, HEADING_PID_KD,
    HEADING_PID_INTEGRAL_LIMIT, HEADING_PID_OUTPUT_LIMIT,
    THRUST_DEADZONE,
    WAYPOINT_MAX_SPEED,
)
from src.utils.geo import heading_error
from src.utils.logger import setup_logger

log = setup_logger("thrust")


@dataclass
class ThrustCommand:
    """High-level thrust vector sent to ESP32."""
    surge: float = 0.0   # Forward (+) / backward (-) thrust [-1, 1]
    sway: float = 0.0    # Right (+) / left (-) thrust [-1, 1]
    yaw: float = 0.0     # Clockwise (+) / counter-clockwise (-) [-1, 1]
    timestamp: float = 0.0


class ThrustComputer:
    """Computes thrust vectors from navigation output + PID controllers.
    
    Two PID controllers:
      1. Position PID -> surge, sway (for station-keeping) or surge (for waypoint nav)
      2. Heading PID -> yaw
    """

    def __init__(self):
        # Position PID (north-south and east-west axes)
        self.pid_north = PIDController(
            kp=POSITION_PID_KP, ki=POSITION_PID_KI, kd=POSITION_PID_KD,
            integral_limit=POSITION_PID_INTEGRAL_LIMIT,
            output_limit=POSITION_PID_OUTPUT_LIMIT,
        )
        self.pid_east = PIDController(
            kp=POSITION_PID_KP, ki=POSITION_PID_KI, kd=POSITION_PID_KD,
            integral_limit=POSITION_PID_INTEGRAL_LIMIT,
            output_limit=POSITION_PID_OUTPUT_LIMIT,
        )

        # Heading PID
        self.pid_heading = PIDController(
            kp=HEADING_PID_KP, ki=HEADING_PID_KI, kd=HEADING_PID_KD,
            integral_limit=HEADING_PID_INTEGRAL_LIMIT,
            output_limit=HEADING_PID_OUTPUT_LIMIT,
        )
        self.pid_heading.set_heading_mode(True)

        self.command = ThrustCommand()

    def compute(self, nav_output: NavigationOutput, state: FusedState) -> ThrustCommand:
        """Compute thrust command from navigation output and current state.
        
        Args:
            nav_output: Output from WaypointNavigator.
            state: Current fused sensor state.
        
        Returns:
            ThrustCommand with surge, sway, yaw in [-1, 1].
        """
        surge = 0.0
        sway = 0.0
        yaw = 0.0

        if nav_output.mode == NavigationState.WAYPOINT_NAV:
            surge, yaw = self._compute_waypoint(nav_output, state)

        elif nav_output.mode == NavigationState.STATION_KEEP:
            surge, sway, yaw = self._compute_station_keeping(nav_output, state)

        elif nav_output.mode == NavigationState.MANUAL:
            # Manual mode: return the command set by set_manual_thrust() directly
            self.command.timestamp = time.time()
            return self.command

        elif nav_output.mode == NavigationState.IDLE:
            # All PIDs reset, zero thrust
            self.pid_north.reset()
            self.pid_east.reset()
            self.pid_heading.reset()
            surge = sway = yaw = 0.0

        # Apply deadzone
        if abs(surge) < THRUST_DEADZONE:
            surge = 0.0
        if abs(sway) < THRUST_DEADZONE:
            sway = 0.0
        if abs(yaw) < THRUST_DEADZONE:
            yaw = 0.0

        self.command = ThrustCommand(
            surge=max(-1.0, min(1.0, surge)),
            sway=max(-1.0, min(1.0, sway)),
            yaw=max(-1.0, min(1.0, yaw)),
            timestamp=time.time(),
        )
        return self.command

    def _compute_waypoint(self, nav: NavigationOutput, state: FusedState) -> tuple:
        """Waypoint following: surge forward along desired heading, PID corrects yaw.
        
        Returns:
            (surge, yaw)
        """
        # Heading PID: correct heading error via yaw
        heading_err = heading_error(state.heading, nav.desired_heading)
        yaw = self.pid_heading.compute(nav.desired_heading, state.heading)

        # Surge: scale desired speed, but reduce when heading error is large
        # (don't push forward when pointing wrong way)
        heading_factor = max(0.0, 1.0 - abs(heading_err) / 90.0)
        surge = nav.desired_speed * heading_factor
        # Normalize surge to [-1, 1] (desired_speed is in m/s, normalize by max)
        if WAYPOINT_MAX_SPEED > 0:
            surge = max(-1.0, min(1.0, surge / WAYPOINT_MAX_SPEED))

        return surge, yaw

    def _compute_station_keeping(self, nav: NavigationOutput, state: FusedState) -> tuple:
        """Dynamic positioning: PID on position error -> surge/sway, PID on heading -> yaw.
        
        Position error is in body frame (north_m, east_m) from navigation module.
        We convert to body-relative surge (forward) and sway (right) using heading.
        
        Returns:
            (surge, sway, yaw)
        """
        # Position PID on NED error
        north_cmd = self.pid_north.compute(0.0, -nav.desired_north_m)  # error = setpoint - measurement
        east_cmd = self.pid_east.compute(0.0, -nav.desired_east_m)

        # Rotate NED commands to body frame using current heading
        heading_rad = math.radians(state.heading)
        surge = north_cmd * math.cos(heading_rad) + east_cmd * math.sin(heading_rad)
        sway = -north_cmd * math.sin(heading_rad) + east_cmd * math.cos(heading_rad)

        # Heading PID: maintain desired heading
        yaw = self.pid_heading.compute(nav.hold_heading, state.heading)

        return surge, sway, yaw

    def reset_all(self):
        """Reset all PID controllers."""
        self.pid_north.reset()
        self.pid_east.reset()
        self.pid_heading.reset()
        self.command = ThrustCommand()
        log.info("All PIDs reset")

    def set_manual_thrust(self, surge: float, sway: float, yaw: float):
        """Set direct thrust for manual mode."""
        self.command = ThrustCommand(
            surge=max(-1.0, min(1.0, surge)),
            sway=max(-1.0, min(1.0, sway)),
            yaw=max(-1.0, min(1.0, yaw)),
            timestamp=time.time(),
        )
