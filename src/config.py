"""
ASV pi-control Configuration
All constants, sensor addresses, PID gains, and system parameters.
"""
from pathlib import Path

# Base directory: ~/pi-controller (works for any user / OS)
BASE_DIR = Path.home() / "pi-controller"

# ---------------------------------------------------------------------------
# Sensor Configuration
# ---------------------------------------------------------------------------

# NEO-6M GPS (pigpio bit-bang UART)
GPS_BB_GPIO = 23              # GPS TX → GPIO23 (pin 16), bit-bang read via pigpiod
GPS_BAUD_RATE = 9600
GPS_TIMEOUT_S = 2.0  # seconds before declaring GPS lost

# MPU6050 IMU (I2C)
MPU6050_I2C_BUS = 1
MPU6050_I2C_ADDR = 0x68
MPU6050_ACCEL_RANGE = 2       # ±2g
MPU6050_GYRO_RANGE = 250      # ±250 deg/s
MPU6050_SAMPLE_RATE_HZ = 50

# GY-273 / QMC5883P Magnetometer (I2C) — board labeled HMC5883L but chip is QMC5883P
HMC5883L_I2C_BUS = 1
HMC5883L_I2C_ADDR = 0x2C
HMC5883L_SAMPLE_RATE_HZ = 15  # HMC5883L max ~75Hz, 15Hz is good for surface
MAGNETIC_DECLINATION_DEG = -1.5  # Adjust for your location (degrees)
HEADING_MOUNT_OFFSET_DEG = 155.0  # Sensor mounted 180° from bow — N↔S swap fix

# ---------------------------------------------------------------------------
# Sensor Fusion
# ---------------------------------------------------------------------------
FUSION_UPDATE_RATE_HZ = 20
# Complementary filter weights (sum to 1.0)
FUSION_GYRO_WEIGHT = 0.98       # Trust gyro for short-term heading changes
FUSION_MAG_WEIGHT = 0.02        # Trust magnetometer for long-term drift correction
FUSION_GPS_POSITION_ALPHA = 0.7 # Low-pass filter coefficient for GPS position

# ---------------------------------------------------------------------------
# PID Controller Gains
# ---------------------------------------------------------------------------

# Position PID (surge + sway) - output: normalized thrust [-1, 1]
POSITION_PID_KP = 0.5
POSITION_PID_KI = 0.05
POSITION_PID_KD = 0.1
POSITION_PID_INTEGRAL_LIMIT = 2.0
POSITION_PID_OUTPUT_LIMIT = 1.0

# Heading PID (yaw) - output: normalized yaw command [-1, 1]
HEADING_PID_KP = 1.0
HEADING_PID_KI = 0.1
HEADING_PID_KD = 0.3
HEADING_PID_INTEGRAL_LIMIT = 1.0
HEADING_PID_OUTPUT_LIMIT = 1.0

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
WAYPOINT_ARRIVAL_RADIUS_M = 2.0       # meters to consider "arrived"
# Measured hull speed (log 23-10-2026: 0.69 m/s peak at surge 1.0).  This is
# also the surge normaliser in thrust.py, so a smaller value gives the boat a
# usable command range instead of 30-50% throttle everywhere.
WAYPOINT_MAX_SPEED = 0.55             # m/s max navigation speed
WAYPOINT_SLOWDOWN_RADIUS_M = 5.0      # start slowing down within this radius
STATION_KEEPING_DEADBAND_M = 0.5      # position deadband for DP mode

# Waypoint surge shaping.  The old rule was surge = 1 - |err|/90, which dropped
# to zero anywhere past 90 deg and left the boat spinning in place.
WAYPOINT_SURGE_FALLOFF_DEG = 150.0    # heading error where surge reaches the floor
WAYPOINT_MIN_SURGE = 0.25             # never command less than this while aimed
WAYPOINT_TURN_IN_PLACE_DEG = 100.0    # beyond this, rotate first and drop surge

# ---------------------------------------------------------------------------
# Thruster Configuration
# ---------------------------------------------------------------------------
# 4 thrusters at 45-degree tilt in X-configuration
#   T_NE (front-right), T_NW (front-left), T_SE (rear-right), T_SW (rear-left)
THRUSTER_COUNT = 4
THRUSTER_TILT_ANGLE_DEG = 45.0
THRUST_DEADZONE = 0.05  # below this value, motor won't spin

# ---------------------------------------------------------------------------
# ESP32 Communication (UART serial — JSON frames over USB/UART)
# ---------------------------------------------------------------------------
ESP32_SERIAL_PORT = "/dev/serial0"  # hardware UART to ESP32 (was /dev/ttyUSB0)
ESP32_BAUD_RATE = 115200
ESP32_HEARTBEAT_INTERVAL_S = 0.5    # Send heartbeat every 500ms
ESP32_TIMEOUT_S = 1.0               # ESP32 failsafe: no msg for 1s -> shutdown motor

# ---------------------------------------------------------------------------
# Firebase Configuration
# ---------------------------------------------------------------------------
FIREBASE_ENABLED = True  # Set True and provide credentials
FIREBASE_CREDENTIALS_PATH = str(BASE_DIR / "firebase-credentials.json")
FIREBASE_DATABASE_URL = "https://autonomous-buoy-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_LOG_RATE_HZ = 2          # Push telemetry at 2Hz
FIREBASE_VEHICLE_ID = "asv-001"   # Unique ID for this vehicle

# Firebase RTDB paths (used by both RPi and React dashboard):
#   RPi writes:  /vehicles/{id}/telemetry/...
#   React writes: /vehicles/{id}/commands/...
#   React reads:  /vehicles/{id}/telemetry/... (listener)
#   RPi reads:    /vehicles/{id}/commands/... (polling)

WEB_DASHBOARD_PORT = 8080               # Port for the web dashboard server

# ---------------------------------------------------------------------------
# ML Inference
# ---------------------------------------------------------------------------
ML_ANOMALY_ENABLED = False
ML_PREDICTION_ENABLED = False
ML_INFERENCE_RATE_HZ = 1          # Run ML models at 1Hz
ML_ANOMALY_HISTORY_SIZE = 500     # Number of samples for Isolation Forest
ML_MODEL_PATH = str(BASE_DIR / "models")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = str(BASE_DIR / "logs" / "asv.log")
LOG_TO_FILE = True
LOG_TO_CONSOLE = True

# ---------------------------------------------------------------------------
# RL Policy (PPO) — deployment safety, not training
# ---------------------------------------------------------------------------
# Every value here is a Pi-side deployment limit. The training repo's own
# safety block (drl/runs/r7_policy_meta.json) is NOT deployment config: the
# Pi-side ControllerMux overrides geofence and watchdog on purpose.
RL_ENABLED = False                    # deployment safety, not training — PID is the default controller
RL_DEFAULT_CONTROLLER = "pid"         # deployment safety, not training — "pid" or "ppo"
RL_MODEL_PATH = str(BASE_DIR / "models" / "policy.onnx")
RL_META_PATH = str(BASE_DIR / "models" / "policy_meta.json")
# The Pi runs from ~/pi-controller (BASE_DIR).  On a dev laptop the model lives in
# the repo, so fall back to it rather than reporting the policy as unavailable.
if not Path(RL_MODEL_PATH).exists():
    _repo_models = Path(__file__).resolve().parent.parent / "models"
    if (_repo_models / "policy.onnx").exists():
        RL_MODEL_PATH = str(_repo_models / "policy.onnx")
        RL_META_PATH = str(_repo_models / "policy_meta.json")
RL_GEOFENCE_M = 20.0                  # deployment safety, not training — policy meta default 80 m must NOT be used for early trials
RL_WATCHDOG_S = 0.5                   # deployment safety, not training — max age of the fused state before PID fallback
RL_MAX_INFERENCE_MS = 50              # deployment safety, not training — slowest acceptable policy step
RL_MAX_THRUST_SCALE = 1.0             # deployment safety, not training — below 1.0 changes behaviour
RL_LOG_DIR = str(BASE_DIR / "logs" / "rl")
# Same dev-laptop fallback as the model paths above: a missing ~/pi-controller
# must not lose the black-box log (the sysid/replay input).
if not Path(BASE_DIR / "models" / "policy.onnx").exists():
    RL_LOG_DIR = str(Path(__file__).resolve().parent.parent / "logs" / "rl")
