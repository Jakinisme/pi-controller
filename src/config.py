"""
ASV pi-control Configuration
All constants, sensor addresses, PID gains, and system parameters.
"""

# ---------------------------------------------------------------------------
# Sensor Configuration
# ---------------------------------------------------------------------------

# NEO-6M GPS (UART)
GPS_SERIAL_PORT = "/dev/serial0"  # or "/dev/ttyS0" on some RPi setups
GPS_BAUD_RATE = 9600
GPS_TIMEOUT_S = 2.0  # seconds before declaring GPS lost

# MPU6050 IMU (I2C)
MPU6050_I2C_BUS = 1
MPU6050_I2C_ADDR = 0x68
MPU6050_ACCEL_RANGE = 2       # ±2g
MPU6050_GYRO_RANGE = 250      # ±250 deg/s
MPU6050_SAMPLE_RATE_HZ = 50

# GY-273 / HMC5883L Magnetometer (I2C)
HMC5883L_I2C_BUS = 1
HMC5883L_I2C_ADDR = 0x1E
HMC5883L_SAMPLE_RATE_HZ = 15  # HMC5883L max ~75Hz, 15Hz is good for surface
MAGNETIC_DECLINATION_DEG = -1.5  # Adjust for your location (degrees)

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
WAYPOINT_MAX_SPEED = 1.5              # m/s max navigation speed
WAYPOINT_SLOWDOWN_RADIUS_M = 5.0      # start slowing down within this radius
STATION_KEEPING_DEADBAND_M = 0.5      # position deadband for DP mode

# ---------------------------------------------------------------------------
# Thruster Configuration
# ---------------------------------------------------------------------------
# 4 thrusters at 45-degree tilt in X-configuration
#   T_NE (front-right), T_NW (front-left), T_SE (rear-right), T_SW (rear-left)
THRUSTER_COUNT = 4
THRUSTER_TILT_ANGLE_DEG = 45.0
THRUST_DEADZONE = 0.05  # below this value, motor won't spin

# ---------------------------------------------------------------------------
# ESP32 Communication
# ---------------------------------------------------------------------------
ESP32_SERIAL_PORT = "/dev/ttyUSB0"  # USB serial to ESP32
ESP32_BAUD_RATE = 115200
ESP32_HEARTBEAT_INTERVAL_S = 0.5    # Send heartbeat every 500ms
ESP32_TIMEOUT_S = 1.0               # ESP32 failsafe if no msg for 1s

# ---------------------------------------------------------------------------
# Firebase Configuration
# ---------------------------------------------------------------------------
FIREBASE_ENABLED = True  # Set True and provide credentials
FIREBASE_CREDENTIALS_PATH = "/home/pi/pi-control/firebase-credentials.json"
FIREBASE_DATABASE_URL = "https://your-project.firebaseio.com"
FIREBASE_LOG_RATE_HZ = 2          # Push telemetry at 2Hz
FIREBASE_VEHICLE_ID = "asv-001"   # Unique ID for this vehicle

# Firebase RTDB paths (used by both RPi and React dashboard):
#   RPi writes:  /vehicles/{id}/telemetry/...
#   React writes: /vehicles/{id}/commands/...
#   React reads:  /vehicles/{id}/telemetry/... (listener)
#   RPi reads:    /vehicles/{id}/commands/... (polling)

# ---------------------------------------------------------------------------
# Camera Configuration (RTSP / HLS Stream)
# ---------------------------------------------------------------------------
CAMERA_ENABLED = True
CAMERA_DEVICE = "/dev/video0"          # USB cam or Pi Cam via libcamera
CAMERA_RESOLUTION = "640x480"          # Lower = less CPU on RPi
CAMERA_FPS = 15                        # Frames per second
CAMERA_BITRATE = "800k"                # Video bitrate
CAMERA_HLS_SEGMENT_TIME = 2            # HLS segment length in seconds
CAMERA_HLS_OUTPUT_DIR = "/tmp/hls"     # Where HLS segments are written
CAMERA_HTTP_PORT = 8554                # HTTP port to serve HLS stream
WEB_DASHBOARD_PORT = 8080               # Port for the web dashboard server
# The stream URL will be: http://<rpi-ip>:8554/stream.m3u8
# This URL is pushed to Firebase so the React dashboard can connect.

# ---------------------------------------------------------------------------
# ML Inference
# ---------------------------------------------------------------------------
ML_ANOMALY_ENABLED = True
ML_PREDICTION_ENABLED = True
ML_INFERENCE_RATE_HZ = 1          # Run ML models at 1Hz
ML_ANOMALY_HISTORY_SIZE = 500     # Number of samples for Isolation Forest
ML_MODEL_PATH = "/home/pi/pi-control/models"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = "/home/pi/pi-control/logs/asv.log"
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
