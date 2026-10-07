# Smart Kitchen (IoT)

[← Back to portfolio root](../README.md)

Small MQTT + Arduino kitchen monitor: gas/PIR/RFID sensors, lock/vent/alarms, MySQL log, and a simple SCADA page.

Not a computer-vision project — kept as a **separate** folder in this repo.

## Pieces

| File | Role |
|------|------|
| `senzorski_arduino_final.txt` | Sensor MCU: MQ-135, PIR, RFID |
| `croduino_final.ino` | ESP8266 UART → MQTT telemetry/events |
| `Aktuatorski_arduino_final.ino` | Actuator MCU: servo lock, vent, buzzer, LCD |
| `actuatord.py` | MQTT `iotcmd` → serial commands |
| `iot_rules.py` | Rules: RFID unlock, PIR auth wait, fire threshold |
| `mqtttodb.py` | MQTT → MySQL |
| `ws_scada.py` | WebSocket SCADA (port 8888) |
| `index.html` | Browser overrides (vent, lock, alarms) |

Set WiFi / MQTT host / DB password locally before running (`YOUR_WIFI_*`, `CHANGE_ME`).
