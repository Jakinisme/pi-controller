"""ASV WAVE-GUARD ML sensor layer (shadow mode).

Lapisan observasi: ekstraksi fitur sensor + logging + telemetri.
Scoring mati permanen sampai model sensor-trained ada (model v2 dilatih
dari proxy/satelit; fitur real_sensor di-exclude training).
Lihat SENSOR_CONTRACT.md untuk kontrak data + skema node Firebase.
"""
