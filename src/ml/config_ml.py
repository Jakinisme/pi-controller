"""Config ML layer — ASV WAVE-GUARD.

Satu tempat untuk semua konstanta ML. Pin/kalibrasi sensor baru = TODO
(verifikasi fisik). Schema training (feature_schema.json) menang; tidak
ada daftar fitur di sini.
"""
import os
from pathlib import Path

# ── path ──────────────────────────────────────────────
# Artifacts training (joblib/climatology/schema) di-copy ke Pi.
# Default: di samping src/ml/ agar bisa di-ship satu paket.
_ML_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = _ML_DIR / "artifacts"
FEATURE_SCHEMA_JSON = ARTIFACTS_DIR / "feature_schema.json"
FEATURE_LIST_JSON = ARTIFACTS_DIR / "feature_list.json"
CLIMATOLOGY_JSON = ARTIFACTS_DIR / "climatology.json"
MODEL_PATH = ARTIFACTS_DIR / "tuban_isolation_forest_v2.joblib"

# Log fitur + ringkasan data mentah (bahan retraining dengan data asli)
LOG_DIR = Path.home() / "pi-controller" / "ml_logs"
LOG_MAX_BYTES = 50 * 1024 * 1024   # 50 MB rotasi (SD card terbatas)
LOG_KEEP = 3                       # simpan 3 file rotasi

# ── mode ──────────────────────────────────────────────
# ML_ENABLED=False: lapisan ML tidak jalan sama sekali (default aman).
# SHADOW_MODE=True: label telemetri tetap non_operational (tidak dipakai
#   kontrol kapal). TIDAK lagi memblokir scoring — lihat SCORING_ENABLED.
ML_ENABLED = True
SHADOW_MODE = True
# Scoring ON: prediksi anomaly (score + is_anomaly) keluar ke telemetri/ml
# dan ml_logs untuk evaluasi. Output TIDAK dipakai menggerakkan aktuator.
# Override tanpa edit file: env ML_SCORING_ENABLED=0|1.
SCORING_ENABLED = os.environ.get("ML_SCORING_ENABLED", "1") == "1"
# Model v2 dilatih dari proxy (aturan 6.6). Bila True, scoring di-skip
# setiap kali pressure/IMU NaN -> prediction tidak pernah keluar di rig
# tanpa sensor itu. False: NaN tetap diimpute climatology, scoring jalan,
# kualitas ditandai lewat degraded/missing_sensors/ood.
SCORING_BLOCK_ON_CORE_MISSING = False

# ── agregasi (konsisten schema training) ──────────────
# PENTING: feature_list.json menang atas nilai di sini (ml_runner baca
# artifact dulu). Artifact training masih 86400/D. Untuk evaluasi cepat,
# override runtime lewat env — tidak perlu edit artifact:
#   ML_AGG_WINDOW_SECONDS=300 python -m src.main
# Nilai window TIDAK mengubah skala fitur (climatology/ma/std tetap sama),
# hanya seberapa cepat satu prediksi keluar. Model tetap dianggap valid;
# longgarkan ke <86400 hanya untuk evaluasi, bukan operasi.
AGG_INTERVAL = "D"             # pandas offset alias; D = harian
AGG_WINDOW_SECONDS = int(os.environ.get("ML_AGG_WINDOW_SECONDS", "300"))
ROLLING_WINDOW = 12            # 12 langkah AGG_INTERVAL

# ── fisika ────────────────────────────────────────────
G = 9.81                       # gravitasi m/s^2
RHO_SEAWATER = 1025.0          # kg/m^3
P_ATM = 101325.0               # Pa rujukan

# ── storm context (verifikasi ke BMKG!) ───────────────
# WIND_STORM_THRESHOLD: Beaufort 8 (Gale) = 17.2 m/s (WMO). Laut Jawa
#   mungkin perlu lebih rendah — cek kategori BMKG perairan Indonesia.
# WIND_STORM_THRESHOLD: Beaufort 6 (10.8-13.8 m/s). Angin nyata Laut Jawa max
# ~9.5 m/s -> Beaufort 8 (17.2) tidak pernah trigger. 10.8 = batas bawah Bft 6.
# HARUS sama dengan config_v2.py (machine-learning) supaya train/serve konsisten.
WIND_STORM_THRESHOLD = 10.8    # m/s (Beaufort 6)
# RAIN_HEAVY_THRESHOLD: BMKG "hujan sangat lebat" >100 mm/hari ~= 4.2 mm/h.
RAIN_HEAVY_THRESHOLD = 5.0     # mm/h mean over AGG_INTERVAL

# ── kualitas IMU ──────────────────────────────────────
# Minimum sampel & rate efektif per WINDOW_SECONDS. Di bawah ambang ->
# fitur NaN + flag, bukan angka ngawur.
IMU_MIN_SAMPLES = 100          # minimal ~1 Hz selama 120s default
IMU_MIN_EFFECTIVE_RATE_HZ = 1.0
IMU_MAX_GAP_S = 2.0            # gap terbesar yang masih ditoleransi

# ── kualitas GPS ──────────────────────────────────────
GPS_MIN_HDOP = 5.0             # di atas ini -> low_confidence (tidak ada HDOP dari NEO-6M GGA? ada, field hdop)
GPS_DRIFT_MIN_M = 2.0          # noise GPS beberapa meter -> ambang minimum

# ── pressure HX710B (pin FINAL — pilih aman, hindari I2C/UART/GPS/SPI/EEPROM) ──
# GPIO 5 (pin 29) & GPIO 6 (pin 31): bebas, tidak ada fungsi alt default.
HX710B_DOUT_GPIO = 5           # pin fisik 29
HX710B_SCK_GPIO = 6            # pin fisik 31
HX710B_SAMPLE_RATE_HZ = 1.0    # mode 10Hz/40Hz? TODO datasheet
# Kalibrasi: offset nol (selang belum tercelup) + scale (counts->kPa).
# TODO 2-titik kalibrasi fisik.
HX710B_OFFSET_COUNTS = 0.0
HX710B_SCALE_KPA_PER_COUNT = 0.001   # placeholder — TODO kalibrasi
HX710B_SATURATION_COUNTS = 0x7FFFFF  # 24-bit two's complement max
TUBE_DEPTH_OFFSET_M = 0.0      # kedalaman ujung selang di bawah lambung
# Rentang sensor 40 kPa ~ 4 m kolom air laut; guard saturasi.
HX710B_MAX_HEAD_M = 4.0

# ── anemometer (ref: flood-sensor/src/anemometer.cpp) ─
# Pulsa reed/Hall RISING. Kalibrasi porting dari ESP32 (TODO verifikasi fisik).
ANEMOMETER_GPIO = 17           # pin fisik 11 (GPIO bebas, aman)
ANEMOMETER_DEBOUNCE_US = 5000  # 5ms (dari ref)
ANEMOMETER_WINDOW_S = 10.0     # RPS = count/window (dari ref: 10s)
# m/s = -0.0181*RPS^2 + 1.3859*RPS + 1.4055 (dari ref)
ANEMOMETER_COEF_A = -0.0181
ANEMOMETER_COEF_B = 1.3859
ANEMOMETER_COEF_C = 1.4055
ANEMOMETER_DEADBAND_MS = 1.5    # <=1.5 m/s -> 0 (dari ref)
ANEMOMETER_LEVEL_SHIFTER_TODO = True   # TODO: sensor 5V? Pi 3.3V — verifikasi

# ── tipping bucket (ref: flood-sensor/src/curah_hujan.cpp) ─
TIPPING_BUCKET_GPIO = 22      # pin fisik 15 (GPIO bebas, aman)
TIPPING_DEBOUNCE_US = 500000   # 500ms (dari ref delay(500)); pigpio glitch filter
RAIN_MM_PER_TIP = 0.70         # dari ref; TODO verifikasi fisik bucket ini

# ── model_feed (gelombang/arus dari WAVERYS/GLORYS NRT) ─
# None = stub. Fitur gelombang/arus NaN -> diimpute climatology, degraded.
# TODO: implementasi fetch NRT butuh kredensial CDS+CMEMS.
MODEL_FEED_ENABLED = False

# ── telemetry ─────────────────────────────────────────
ML_TELEMETRY_INTERVAL_S = AGG_WINDOW_SECONDS   # per AGG_INTERVAL, bukan per sampel
