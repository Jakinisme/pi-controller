"""
Generic PID Controller with anti-windup, derivative filtering, and output clamping.
"""

import time
from dataclasses import dataclass, field


@dataclass
class PIDGains:
    """PID gain parameters."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0


class PIDController:
    """PID controller with anti-windup and derivative low-pass filter.
    
    Usage:
        pid = PIDController(kp=1.0, ki=0.1, kd=0.05, output_limit=1.0)
        output = pid.compute(setpoint=100.0, measurement=95.0)
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        integral_limit: float = 100.0,
        output_limit: float = 1.0,
        derivative_filter_coeff: float = 0.1,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral_limit = integral_limit
        self._output_limit = output_limit
        self._d_filter = derivative_filter_coeff  # Low-pass on derivative

        # Internal state
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0
        self._last_time = 0.0
        self._is_heading = False  # Special mode for heading wrap-around

    def set_gains(self, kp: float, ki: float, kd: float):
        """Update PID gains at runtime (for tuning via dashboard)."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_heading_mode(self, enabled: bool = True):
        """Enable heading mode: error wraps around [-180, 180] degrees."""
        self._is_heading = enabled

    def reset(self):
        """Reset internal state (call when switching modes)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0
        self._last_time = 0.0

    def compute(self, setpoint: float, measurement: float, dt: float = 0.0) -> float:
        """Compute PID output.
        
        Args:
            setpoint: Desired value.
            measurement: Current measured value.
            dt: Time step in seconds. If 0, computed from wall clock.
        
        Returns:
            Control output, clamped to [-output_limit, +output_limit].
        """
        now = time.time()
        if dt <= 0:
            dt = now - self._last_time if self._last_time > 0 else 0.05
        self._last_time = now
        dt = max(dt, 0.001)  # Prevent division by zero
        dt = min(dt, 0.5)    # Prevent huge dt spikes

        # Error calculation
        error = setpoint - measurement
        if self._is_heading:
            # Wrap heading error to [-180, 180]
            error = (error + 180) % 360 - 180

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup (clamping)
        self._integral += error * dt
        self._integral = max(
            -self._integral_limit,
            min(self._integral_limit, self._integral)
        )
        i_term = self.ki * self._integral

        # Derivative with low-pass filter (reduces noise)
        raw_derivative = (error - self._prev_error) / dt
        filtered_derivative = (
            self._d_filter * raw_derivative
            + (1 - self._d_filter) * self._prev_derivative
        )
        d_term = self.kd * filtered_derivative
        self._prev_derivative = filtered_derivative
        self._prev_error = error

        # Total output with clamping
        output = p_term + i_term + d_term
        output = max(-self._output_limit, min(self._output_limit, output))
        return output

    @property
    def integral(self) -> float:
        return self._integral

    @property
    def last_error(self) -> float:
        return self._prev_error
