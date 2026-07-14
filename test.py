"""
ASV Test Suite
"""
import sys
import os

# Inject hardware mock before importing anything from src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from hardware_mock import install
install()

import pytest
import math
from src.utils.geo import haversine_distance, bearing, heading_error
from src.control.pid import PIDController
from src.control.navigation import WaypointNavigator, Waypoint, NavigationState, NavigationOutput
from src.control.thrust import ThrustComputer
from src.sensors.fusion import FusedState, SensorFusion
from src.sensors.gps import GPSReader

class TestGeo:
    def test_haversine_same_point(self):
        lat, lon = -6.8969, 112.0521
        assert haversine_distance(lat, lon, lat, lon) == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        lat1, lon1 = -6.8969, 112.0521
        lat2, lon2 = -6.8960, 112.0521  # ~100m utara
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        assert 90 < dist < 120

    def test_bearing_north(self):
        assert bearing(-6.9, 112.0, -6.8, 112.0) == pytest.approx(0.0, abs=1.0)

    def test_bearing_east(self):
        assert bearing(-6.9, 112.0, -6.9, 112.1) == pytest.approx(90.0, abs=1.0)

    def test_heading_error(self):
        assert heading_error(350, 10) == pytest.approx(20.0)
        assert heading_error(10, 350) == pytest.approx(-20.0)
        assert abs(heading_error(90, 270)) == pytest.approx(180.0)

class TestPID:
    def test_proportional(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limit=1.0)
        out = pid.compute(setpoint=1.0, measurement=0.5, dt=0.1)
        assert out == pytest.approx(0.5)

    def test_output_clamped(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=1.0)
        out = pid.compute(setpoint=10.0, measurement=0.0, dt=0.1)
        assert out <= 1.0

    def test_heading_mode(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limit=180.0)
        pid.set_heading_mode(True)
        # target=10, current=350 -> error=+20
        out = pid.compute(setpoint=10.0, measurement=350.0, dt=0.1)
        assert out == pytest.approx(20.0)

class TestNavigation:
    def test_waypoint_nav_arrival(self):
        nav = WaypointNavigator(arrival_radius=3.0)
        wps = [Waypoint(-6.8969, 112.0521)]
        nav.start_waypoint_nav(wps)
        
        # Cukup dekat (0.5m)
        out = nav.update(-6.89690, 112.052104, heading=0.0)
        # Harusnya nyampe dan langsung IDLE (karena ini waypoint terakhir)
        assert nav.state == NavigationState.IDLE

    def test_station_keeping(self):
        nav = WaypointNavigator()
        nav.start_station_keeping(lat=-6.8969, lon=112.0521, heading=45.0)
        assert nav.state == NavigationState.STATION_KEEP
        
        out = nav.update(-6.8970, 112.0521, heading=45.0)
        assert out.mode == NavigationState.STATION_KEEP
        assert out.hold_heading == 45.0
        assert out.desired_north_m != 0.0

class TestBlendAngle:
    def test_wraparound_near_north(self):
        # 359° and 1° are 2° apart; plain average would yield ~351°
        result = SensorFusion._blend_angle(359, 1, 0.02)
        assert result == pytest.approx(359.04, abs=0.01)
        assert result > 358 or result < 2

    def test_wraparound_halfway_shortest_path(self):
        # 10° and 350° are 20° apart via north; midpoint is 0°, not 180°
        result = SensorFusion._blend_angle(10, 350, 0.5)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_weight_zero_returns_base(self):
        assert SensorFusion._blend_angle(45.0, 90.0, 0.0) == pytest.approx(45.0)

    def test_weight_one_returns_target(self):
        assert SensorFusion._blend_angle(45.0, 90.0, 1.0) == pytest.approx(90.0)

class TestThrustComputer:
    def test_idle(self):
        tc = ThrustComputer()
        nav_out = NavigationOutput(mode=NavigationState.IDLE)
        cmd = tc.compute(nav_out, FusedState())
        assert cmd.surge == 0.0
        assert cmd.sway == 0.0
        assert cmd.yaw == 0.0

    def test_manual(self):
        tc = ThrustComputer()
        tc.set_manual_thrust(0.5, -0.5, 0.2)
        nav_out = NavigationOutput(mode=NavigationState.MANUAL)
        cmd = tc.compute(nav_out, FusedState())
        assert cmd.surge == 0.5
        assert cmd.sway == -0.5
        assert cmd.yaw == 0.2

    def test_waypoint_thrust(self):
        tc = ThrustComputer()
        # desired heading = 90, current heading = 90 -> yaw=0
        # desired speed = 1.0 m/s
        nav_out = NavigationOutput(mode=NavigationState.WAYPOINT_NAV, desired_speed=1.0, desired_heading=90.0)
        state = FusedState(heading=90.0)
        cmd = tc.compute(nav_out, state)
        
        assert cmd.surge > 0.0
        assert cmd.sway == 0.0
        assert abs(cmd.yaw) < 0.1

if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])