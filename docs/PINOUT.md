# PINOUT — ASV WAVE-GUARD

Dokumen ini daftar **SEMUA** pin/GPIO terpakai di dua board (Raspberry Pi 3B +
ESP32). Dua sisi terpisah — jangan tertukar nomor GPIO-nya.

Status: **pin sensor ML baru (pressure/anemometer/tipping) BELUM di-assign** —
menunggu wiring fisik. Lihat §4 untuk usulan.

---

## 1. Raspberry Pi 3B — Header 40 pin

Pi jalankan: IMU, magnetometer, GPS, link serial ke ESP32, web dashboard.

| Fungsi | GPIO (BCM) | Pin fisik | Bus/Protokol | Alamat / Catatan |
|---|---|---|---|---|
| SDA | GPIO 2 | **3** | I2C bus 1 | IMU MPU6050 + Magnetometer QMC5883P |
| SCL | GPIO 3 | **5** | I2C bus 1 | idem |
| 3V3 | — | 1, 17 | power | ke IMU + mag |
| 5V | — | 2, 4 | power | ke sensor 5V (butuh level shifter untuk sinyal) |
| GND | — | 6, 9, 14, 20, 25, 30, 34, 39 | ground | common ground semua sensor |
| GPS TX (bit-bang) | **GPIO 23** | **16** | pigpio bit-bang UART | `GPS_BB_GPIO=23` di `src/config.py` |
| UART TXD → ESP32 | GPIO 14 | **8** | `/dev/serial0` | `ESP32_SERIAL_PORT` |
| UART RXD ← ESP32 | GPIO 15 | **10** | `/dev/serial0` | 115200 baud |

### I2C — alamat perangkat (bus 1)

| Chip | Alamat I2C | Fungsi | Config |
|---|---|---|---|
| MPU6050 | `0x68` | IMU (accel/gyro/roll/pitch) | `MPU6050_I2C_ADDR` |
| QMC5883P | `0x2C` | Magnetometer (heading) | `HMC5883L_I2C_ADDR` |

> **Catatan chip:** board magnetometer berlabel "HMC5883L" tapi chip sebenarnya
> **QMC5883P**. Register beda — kode sudah disesuaikan.
> IMU board umumnya MPU6050 (spec awal bilang MPU6500 — ikuti chip di kode).

---

## 2. ESP32 — Thrust Controller

ESP32 jalankan: 4 ESC motor (BLHeli_S, PWM 50 Hz), link serial ke Pi.

| Fungsi | GPIO | Posisi | Catatan |
|---|---|---|---|
| Motor **NW** | **GPIO 25** | front-left | `PIN_MOTOR_NW` di `firmware-esp/src/header/config.h` |
| Motor **NE** | **GPIO 26** | front-right | `PIN_MOTOR_NE` |
| Motor **SW** | **GPIO 27** | rear-left | `PIN_MOTOR_SW` |
| Motor **SE** | **GPIO 14** | rear-right | `PIN_MOTOR_SE` |
| Status LED | GPIO 2 | built-in DevKit | `PIN_LED` |
| UART TX → Pi | GPIO 1 (default) | default UART0 | 115200 baud |
| UART RX ← Pi | GPIO 3 (default) | default UART0 | 115200 baud |
| GND | GND | — | **wajib common ground dengan Pi** |

**Layout thruster (tampak atas, haluan = atas):**
```
        ──── hull kiri ────
       [NW]              [SW]
       front-left        rear-left

              [CENTER]

       [NE]              [SE]
       front-right       rear-right
        ──── hull kanan ────
```

**PWM ESC (BLHeli_S 3D mode, 50 Hz):** 1200 µs = full reverse · 1500 µs = netral ·
1800 µs = full forward. Motor mulai berputar ~1520 µs (maju) / ~1480 µs (mundur).
Failsafe: tidak ada perintah >2000 ms → motor idle.

---

## 3. Ringkasan pin terpakai

### Pi 3B
```
GPIO 2   I2C SDA      (pin 3)
GPIO 3   I2C SCL      (pin 5)
GPIO 14  UART TXD     (pin 8)   -> ESP32 RX
GPIO 15  UART RXD     (pin 10)  <- ESP32 TX
GPIO 23  GPS bit-bang (pin 16)
```

### ESP32
```
GPIO 1   UART TX      -> Pi RX
GPIO 2   LED built-in
GPIO 3   UART RX      <- Pi TX
GPIO 14  Motor SE     (rear-right)
GPIO 25  Motor NW     (front-left)
GPIO 26  Motor NE     (front-right)
GPIO 27  Motor SW     (rear-left)
```

---

## 4. Sensor ML baru — PIN FINAL (isi `src/ml/config_ml.py`)

Tiga sensor baru untuk fitur ML. **Pin di Pi, BUKAN pin ESP32** (pin referensi
proyek flood-sensor itu ESP32: anemo=19, rain=17 — jangan dipakai di Pi).

| Sensor | Sinyal | GPIO Pi | Pin fisik | Config | Status |
|---|---|---|---|---|---|
| HX710B (pressure) | DOUT | **GPIO 5** | 29 | `HX710B_DOUT_GPIO` | FINAL |
| HX710B (pressure) | SCK | **GPIO 6** | 31 | `HX710B_SCK_GPIO` | FINAL |
| Anemometer | pulsa (reed/Hall) | **GPIO 17** | 11 | `ANEMOMETER_GPIO` | FINAL |
| Tipping bucket | pulsa (reed) | **GPIO 22** | 15 | `TIPPING_BUCKET_GPIO` | FINAL |

Dipilih aman: hindari GPIO 2/3 (I2C), 14/15 (UART), 23 (GPS), 0/1 (EEPROM ID),
9/10/11 (SPI + pull resistor), 18/19 (PWM audio default). Semua pin di atas bebas
tanpa fungsi alt default. **Wiring ikuti tabel ini.**

### Peringatan wiring

1. **Level tegangan.** Anemometer referensi & tipping bucket sering 5 V. Pi GPIO
   **3.3 V only** — sinyal 5 V bisa merusak pin. Pakai **level shifter** atau
   pembagi tegangan. Flag: `ANEMOMETER_LEVEL_SHIFTER_TODO = True`.
2. **Pull-up.** Reed/Hall butuh pull-up (Pi punya internal; pastikan di-enable).
3. **HX710B.** 24-bit ADC, protokol DOUT+SCK. DOUT low = data siap; SCK >60 µs =
   power-down. DOUT = data ke Pi (input), SCK = clock dari Pi (output).
4. **Common ground.** Semua sensor + ESP32 + Pi harus satu GND.
5. **pigpio.** Driver pakai pigpiod (bukan RPi.GPIO). Glitch filter untuk debounce;
   **jangan** `delay()` blocking di Pi.

### Kalibrasi (belum diverifikasi fisik)

| Item | Nilai | Sumber |
|---|---|---|
| Anemometer | m/s = −0.0181·RPS² + 1.3859·RPS + 1.4055 | referensi flood-sensor |
| Anemometer deadband | ≤ 1.5 m/s → 0 | referensi |
| Anemometer window | 10 s (RPS = count/10s) | referensi |
| Tipping bucket | 0.70 mm/tip | referensi |
| HX710B offset | 0.0 counts | **placeholder — TODO 2-titik** |
| HX710B scale | 0.001 kPa/count | **placeholder — TODO 2-titik** |
| Kedalaman ujung selang | 0.0 m | TODO ukur |

---

## 5. Sisa TODO setelah wiring

Pin sudah final (§4) — **tinggal wiring fisik ikut tabel**. Yang masih perlu
diukur/diisi setelah hardware nyata terpasang:

```
TUBE_DEPTH_OFFSET_M      = ?    (kedalaman ujung selang di bawah permukaan air)
HX710B_OFFSET_COUNTS     = ?    (kalibrasi 2-titik: selang kering -> 0)
HX710B_KPA_PER_COUNT     = ?    (kalibrasi 2-titik: kedalaman air diketahui)
ANEMOMETER_LEVEL_SHIFTER = ?    (True kalau sensor anemometer 5V)
RAIN_MM_PER_TIP          = 0.70 (verifikasi bucket fisik)
```

Verifikasi pin motor ESP32 (25/26/27/14) cocok dengan wiring-mu.
