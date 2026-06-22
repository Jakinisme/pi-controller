"""
ASV Test Suite
==============
Mirip Jest di React, tapi versi Python pakai pytest.

Install dulu:
    pip install pytest pytest-asyncio --break-system-packages

Jalankan:
    pytest test_asv.py -v
    pytest test_asv.py -v -k "geo"        # test geo aja
    pytest test_asv.py -v -k "pid"        # test pid aja
"""

import pytest
import asyncio
import math
import time
from unittest.mock import MagicMock, patch, AsyncMock

# ─────────────────────────────────────────────
# 1. GEO UTILS  (haversine, bearing, cross-track)
#    Sesuaikan import path dengan struktur proyekmu
# ─────────────────────────────────────────────
# from src.geo import haversine_distance, bearing_to, cross_track_error

# === MOCK dulu kalau belum bisa import ===
def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine formula — returns distance in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing_to(lat1, lon1, lat2, lon2):
    """Bearing dari titik 1 ke titik 2, dalam derajat [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
# ==========================================


class TestGeo:
    """Mirip describe('geo', () => { ... }) di Jest"""

    def test_haversine_same_point(self):
        """Jarak ke titik yang sama harus 0."""
        lat, lon = -6.8969, 112.0521  # Tuban
        dist = haversine_distance(lat, lon, lat, lon)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        """
        Tuban → 100m ke utara (approx).
        ~0.001 derajat latitude ≈ 111 meter.
        """
        lat1, lon1 = -6.8969, 112.0521
        lat2, lon2 = -6.8960, 112.0521  # ~100m ke utara
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        assert 90 < dist < 120, f"Expected ~100m, got {dist:.1f}m"

    def test_haversine_symmetry(self):
        """Jarak A→B == B→A."""
        a = (-6.8969, 112.0521)
        b = (-6.9000, 112.0600)
        assert haversine_distance(*a, *b) == pytest.approx(
            haversine_distance(*b, *a), rel=1e-6
        )

    def test_bearing_north(self):
        """Titik tepat di utara → bearing harus ~0°."""
        lat1, lon1 = -6.9, 112.0
        lat2, lon2 = -6.8, 112.0  # utara
        b = bearing_to(lat1, lon1, lat2, lon2)
        assert b == pytest.approx(0.0, abs=1.0)

    def test_bearing_east(self):
        """Titik tepat di timur → bearing harus ~90°."""
        lat1, lon1 = -6.9, 112.0
        lat2, lon2 = -6.9, 112.1  # timur
        b = bearing_to(lat1, lon1, lat2, lon2)
        assert b == pytest.approx(90.0, abs=1.0)

    def test_bearing_south(self):
        """Titik tepat di selatan → bearing harus ~180°."""
        lat1, lon1 = -6.9, 112.0
        lat2, lon2 = -7.0, 112.0  # selatan
        b = bearing_to(lat1, lon1, lat2, lon2)
        assert b == pytest.approx(180.0, abs=1.0)

    def test_bearing_range(self):
        """Bearing selalu dalam range [0, 360)."""
        coords = [
            ((-6.9, 112.0), (-7.0, 111.9)),
            ((-6.9, 112.0), (-6.8, 112.1)),
            ((-7.0, 113.0), (-6.5, 111.0)),
        ]
        for (p1, p2) in coords:
            b = bearing_to(*p1, *p2)
            assert 0 <= b < 360, f"Bearing out of range: {b}"


# ─────────────────────────────────────────────
# 2. PID CONTROLLER
# ─────────────────────────────────────────────
class PIDController:
    """
    Paste class PID-mu di sini kalau belum bisa import,
    atau uncomment: from src.control.pid import PIDController
    """
    def __init__(self, kp, ki, kd, output_limit=1.0, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit or output_limit
        self._integral = 0.0
        self._last_error = None
        self._last_time = None

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        p = self.kp * error
        self._integral += error * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))
        i = self.ki * self._integral
        d = 0.0
        if self._last_error is not None:
            d = self.kd * (error - self._last_error) / dt
        self._last_error = error
        output = p + i + d
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._integral = 0.0
        self._last_error = None


class TestPID:
    """Test PID Controller — mirip unit test komponen React."""

    def test_proportional_only(self):
        """Ki=0, Kd=0 → output = Kp * error saja."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        out = pid.compute(error=0.5, dt=0.1)
        assert out == pytest.approx(0.5, rel=1e-4)

    def test_zero_error(self):
        """Error 0 → output harus 0 (tanpa integral build-up)."""
        pid = PIDController(kp=1.0, ki=1.0, kd=0.5)
        out = pid.compute(error=0.0, dt=0.1)
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_output_clamped(self):
        """Output tidak boleh melebihi output_limit."""
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=1.0)
        out = pid.compute(error=999.0, dt=0.1)
        assert out <= 1.0

        out_neg = pid.compute(error=-999.0, dt=0.1)
        assert out_neg >= -1.0

    def test_integral_windup_prevention(self):
        """Integral tidak boleh meledak meski error terus ada."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0,
                            output_limit=1.0, integral_limit=1.0)
        for _ in range(1000):
            out = pid.compute(error=1.0, dt=0.1)
        assert out <= 1.0, "Integral windup terjadi!"

    def test_reset(self):
        """Setelah reset(), integral bersih dan mulai fresh."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        for _ in range(10):
            pid.compute(error=1.0, dt=0.1)
        pid.reset()
        out = pid.compute(error=0.1, dt=0.1)
        # Setelah reset, output hanya dari satu step integral kecil
        assert abs(out) < 0.1

    def test_derivative_damps_fast_change(self):
        """Kd harus hasilkan output berlawanan saat error tiba-tiba naik."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        pid.compute(error=0.0, dt=0.1)   # init last_error
        out = pid.compute(error=1.0, dt=0.1)  # error naik → d positif
        assert out > 0


# ─────────────────────────────────────────────
# 3. WAYPOINT NAVIGATION LOGIC
# ─────────────────────────────────────────────
class TestWaypointNavigation:
    """
    Test logic navigasi waypoint.
    Ganti dengan import actual class-mu kalau sudah siap.
    """

    def test_waypoint_reached_within_radius(self):
        """ASV dianggap reach waypoint kalau jarak < threshold."""
        THRESHOLD_M = 3.0
        target_lat, target_lon = -6.8969, 112.0521

        # Simulasi ASV sudah sangat dekat (0.5m)
        current_lat, current_lon = -6.89690, 112.052104
        dist = haversine_distance(current_lat, current_lon, target_lat, target_lon)
        assert dist < THRESHOLD_M, "Seharusnya dianggap waypoint reached"

    def test_waypoint_not_reached_far(self):
        """ASV masih jauh → belum reach waypoint."""
        THRESHOLD_M = 3.0
        target_lat, target_lon = -6.8969, 112.0521

        # ASV masih ~100m jauh
        current_lat, current_lon = -6.8960, 112.0521
        dist = haversine_distance(current_lat, current_lon, target_lat, target_lon)
        assert dist >= THRESHOLD_M, "Seharusnya belum reach waypoint"

    def test_heading_error_wrap(self):
        """
        Heading error harus wrap ke [-180, 180].
        Misal: target 10°, heading 350° → error = -20° (putar kiri dikit)
        bukan +340° (muter hampir penuh ke kanan).
        """
        def heading_error(target, current):
            err = target - current
            # Wrap ke [-180, 180]
            while err > 180:  err -= 360
            while err < -180: err += 360
            return err

        # Dari 350° ke 10°: putar kanan 20° (searah jarum jam) → +20°
        assert heading_error(10, 350)  == pytest.approx(20.0, abs=0.1)
        # Dari 10° ke 350°: putar kiri 20° (berlawanan jarum jam) → -20°
        assert heading_error(350, 10)  == pytest.approx(-20.0, abs=0.1)
        # Tepat berlawanan → bisa +180 atau -180
        assert abs(heading_error(90, 270)) == pytest.approx(180.0, abs=0.1)
        assert heading_error(0, 0)     == pytest.approx(0.0)

    def test_waypoint_list_sequential(self):
        """Setelah reach waypoint pertama, pindah ke berikutnya."""
        waypoints = [
            (-6.8969, 112.0521),
            (-6.8960, 112.0530),
            (-6.8950, 112.0540),
        ]
        current_index = 0
        THRESHOLD = 3.0

        # Simulasi ASV sudah sampai waypoint 0
        asv_lat, asv_lon = waypoints[0]
        dist = haversine_distance(asv_lat, asv_lon, *waypoints[current_index])
        if dist < THRESHOLD:
            current_index += 1

        assert current_index == 1, "Harus pindah ke waypoint berikutnya"


# ─────────────────────────────────────────────
# 4. THRUST MAPPING
# ─────────────────────────────────────────────
class TestThrustMapping:
    """
    Test konversi surge/sway/yaw → PWM tiap thruster.
    Konfigurasi X/pinwheel 45° dengan 4 thruster.
    """

    def _compute_pwm(self, surge, sway, yaw, pwm_min=1100, pwm_max=1900):
        """
        Simplified 4-thruster X-config mixer.
        FL=front-left, FR=front-right, RL=rear-left, RR=rear-right
        Semua di 45°, jadi contribution surge/sway sama untuk tiap motor.
        """
        factor = 1 / math.sqrt(2)
        FL = ( surge * factor - sway * factor + yaw)
        FR = ( surge * factor + sway * factor - yaw)
        RL = ( surge * factor + sway * factor + yaw)
        RR = ( surge * factor - sway * factor - yaw)

        # Normalize ke [-1, 1]
        max_val = max(abs(FL), abs(FR), abs(RL), abs(RR), 1.0)
        FL, FR, RL, RR = FL/max_val, FR/max_val, RL/max_val, RR/max_val

        # Map ke PWM
        center = (pwm_min + pwm_max) / 2
        half   = (pwm_max - pwm_min) / 2
        return {
            'FL': int(center + FL * half),
            'FR': int(center + FR * half),
            'RL': int(center + RL * half),
            'RR': int(center + RR * half),
        }

    def test_idle_is_center_pwm(self):
        """Surge=0, sway=0, yaw=0 → semua thruster PWM = 1500."""
        pwm = self._compute_pwm(surge=0, sway=0, yaw=0)
        for motor, val in pwm.items():
            assert val == pytest.approx(1500, abs=5), \
                f"{motor} idle PWM salah: {val}"

    def test_full_forward(self):
        """Surge=1 → semua thruster harus maju (PWM > 1500)."""
        pwm = self._compute_pwm(surge=1.0, sway=0, yaw=0)
        for motor, val in pwm.items():
            assert val > 1500, f"{motor} harus forward tapi {val}"

    def test_full_reverse(self):
        """Surge=-1 → semua thruster mundur (PWM < 1500)."""
        pwm = self._compute_pwm(surge=-1.0, sway=0, yaw=0)
        for motor, val in pwm.items():
            assert val < 1500, f"{motor} harus reverse tapi {val}"

    def test_pwm_always_in_range(self):
        """PWM tidak boleh keluar dari [1100, 1900] apapun inputnya."""
        test_cases = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
            (1, 1, 1), (-1, -1, -1),
        ]
        for surge, sway, yaw in test_cases:
            pwm = self._compute_pwm(surge, sway, yaw)
            for motor, val in pwm.items():
                assert 1100 <= val <= 1900, \
                    f"PWM out of range [{surge},{sway},{yaw}] {motor}={val}"

    def test_yaw_right_differential(self):
        """
        Yaw positif (belok kanan) → FL & RL lebih kencang dari FR & RR.
        """
        pwm = self._compute_pwm(surge=0, sway=0, yaw=1.0)
        assert pwm['FL'] > pwm['FR'], "FL harus lebih kencang saat yaw kanan"
        assert pwm['RL'] > pwm['RR'], "RL harus lebih kencang saat yaw kanan"


# ─────────────────────────────────────────────
# 5. SENSOR MOCK TEST (async)
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestSensorMock:
    """
    Mock sensor tanpa hardware fisik.
    Mirip jest.mock() di React testing.
    """

    async def test_gps_mock_data(self):
        """Simulasi GPS data valid masuk ke sistem."""
        mock_gps = MagicMock()
        mock_gps.data.latitude  = -6.8969
        mock_gps.data.longitude = 112.0521
        mock_gps.data.fix       = True
        mock_gps.data.timestamp = time.time()

        assert mock_gps.data.fix is True
        assert -90 <= mock_gps.data.latitude  <= 90
        assert -180 <= mock_gps.data.longitude <= 180

    async def test_imu_mock_data(self):
        """Simulasi IMU data valid."""
        mock_imu = MagicMock()
        mock_imu.data.roll    = 2.5   # derajat
        mock_imu.data.pitch   = -1.0
        mock_imu.data.heading = 45.0

        assert -180 <= mock_imu.data.roll    <= 180
        assert -90  <= mock_imu.data.pitch   <= 90
        assert 0    <= mock_imu.data.heading < 360


# ─────────────────────────────────────────────
# ENTRY POINT (jalankan langsung tanpa pytest)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])