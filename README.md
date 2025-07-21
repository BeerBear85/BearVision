BearVision

Instant, hands-free wakeboard highlight clips – no cameraman required.

BearTag ─▶ BLE ─▶ Edge Cam ─▶ 4G/Wi-Fi ─▶ Cloud ─▶ Mobile/Web App

How It Works

1. Tag detected – RSSI spike tells the edge camera to buffer 10 s before the trick.


2. Computer vision crops, tracks and trims the clip.


3. Upload & process – clip is stabilised and pushed to the rider’s phone within seconds.



Tech Stack (MVP)

BearTag – rugged BLE tracker with accelerometer.

Edge unit – Raspberry Pi 4 + wide-angle camera (CV-only mode)【citation RPi CPU?】

Cloud – S3-compatible storage, serverless processing, REST/WebSocket API.

App – React Native / PWA.


Tested at ~2-3 FPS on a Pi 4 CPU-only build

Quick Start (Edge)

git clone https://github.com/your-org/bearvision-edge
cd bearvision-edge && ./setup.sh
python run.py                 # starts live pipeline


