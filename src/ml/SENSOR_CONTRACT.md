# SENSOR_CONTRACT — ASV WAVE-GUARD ML Layer

Kontrak data antara pembaca sensor (live/replay/synthetic) dan lapisan ML.
Schema training (`feature_schema.json`) menang; ini dokumen manusia.

## 1. Sensor per pipeline

| Sensor | Pembaca | Rate | Satuan | Status |
|---|---|---|---|---|
| IMU MPU6050 | `sensors/imu.py` (Pi, I2C 0x68) | 50Hz | accel m/s², gyro deg/s, roll/pitch deg | **ada** — raw diekspos `imu.data`, fan-out read-only |
| GPS NEO-6M | `sensors/gps.py` (Pi, pigpio GPIO23) | ~10Hz poll | lat/lon°, spd m/s, course°, hdop | **ada** |
| Magnetometer QMC5883P | `sensors/magnetometer.py` | 15Hz | heading° | **ada** (heading only, tidak masuk ML) |
| Pressure HX710B | `ml/drivers/pressure.py` (TODO) | 1Hz | kPa -> m head | **stub** — pin/kalibrasi TODO |
| Anemometer | `ml/drivers/anemometer.py` (TODO) | 10s window | m/s | **stub** — pin TODO, kalibrasi dari ref flood-sensor |
| Tipping bucket | `ml/drivers/tipping.py` (TODO) | event | mm, mm/h | **stub** — pin TODO, mm_per_tip=0.70 dari ref |
| Wave/arus | model_feed NRT (WAVERYS/GLORYS) | harian | m, s, m/s | **stub** (None -> imputed, degraded) |

## 2. Dataclass sampel (`sensor_contract.py`)

`SensorSample` base: `timestamp_mono` (time.monotonic), `timestamp_wall`
(time.time()), `ntp_synced`, `valid`, `quality`. Subclass: `IMUSample`,
`GPSSample`, `PressureSample`, `WindSample`, `RainSample`.

**Window dihitung dengan monotonic.** Wall-clock hanya label; Pi tidak punya
RTC. `ntp_synced=False` sampai NTP konvergen — telemetri menandai ini.

## 3. Ring buffer (`buffers.py`)

`RingBuffer(name, window_seconds, max_rate_hz)`. Kapasitas dalam detik.
`append()` non-blocking (lock singkat + trim by time). `snapshot()` copy
read-only. Satu pembaca per sensor; data di-fan-out ke konsumen (heading + ML).

## 4. Fitur (`feature_schema.json` — source of truth)

30 `feature_columns` + 2 `training_excluded` (`gps_heading_change`,
`pressure_level`). AGG_INTERVAL=`D` (86400s), ROLLING_WINDOW=12.
Unit: `imu_accel_rms` m/s², `imu_roll_rms/pitch_rms` **rad**, `imu_dominant_freq` Hz,
`gps_drift_speed` m/s, `gps_drift_distance` m, `pressure_level` m.
**Jangan duplikasi daftar fitur di kode** — baca dari schema.

## 5. Skema node Firebase (`/vehicles/{id}/telemetry/ml`)

Dikirim per AGG_INTERVAL (harian), bukan per sampel, lewat `firebase_bridge.push_ml()`.

```json
{
  "score": <float|null>,
  "is_anomaly": <bool|null>,
  "interpretation": "shadow_no_scoring|normal|anomali_dijelaskan_badai|anomali_cuaca_tenang|sensor_core_hilang",
  "degraded": <bool>,
  "missing_sensors": ["pressure","imu","gps","wind","rain","wave_feed","currents"],
  "storm_context": <bool>,
  "ood": <bool>,
  "non_operational": true,
  "scoring_active": false,
  "model_version": "v2",
  "ts": <unix>
}
```

Dashboard React badge: `non_operational` (shadow), `degraded`, `ood`,
`scoring_active`. **Scoring mati permanen** sampai model sensor-trained ada
(aturan 6.6: model v2 dilatih dari proxy/satelit).

## 6. Referensi kalibrasi sensor baru

Dari `E:\Zaky-Sekolah\UGM\src\flood-sensorr\flood-sensor\src` (ESP32 Arduino):
- **Anemometer** (`anemometer.cpp`): RISING pulse, debounce 5ms, RPS=count/10s,
  `m/s = -0.0181·RPS² + 1.3859·RPS + 1.4055`, deadband ≤1.5→0. Pin ESP32 GPIO19.
- **Tipping bucket** (`curah_hujan.cpp`): FALLING INPUT_PULLUP,
  `MILIMETER_PER_TIP=0.70`, debounce 500ms. Pin ESP32 GPIO17.
- **HX710B**: 24-bit two's complement, DOUT+SCK, SCK>60µs=power-down.
  Kalibrasi offset nol + scale 2-titik. **Belum diverifikasi fisik.**

Pin Pi berbeda — set di `config_ml.py` setelah wiring diverifikasi.
TODO level shifter kalau sensor 5V (Pi 3.3V).
