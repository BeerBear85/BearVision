# BearVision – LLM Context Doc v1.2
**Date:** 25‑07‑2025  
**Version:** 1.2  
**Author:** Mr. Bear (Bjørn Eskildsen)

---

## Purpose
This document gives Large Language Models (LLMs) a concise, unambiguous domain overview of the BearVision project so the model can answer questions, propose improvements, and generate code/architectures consistently.

---

## 1. High‑level architecture
```
BearTag/TrickTag ───▶ BLE ───▶ EDGE Unit = BLE + GoPro control + motion detection + clip tagging + processing ───▶ 4G/Wi‑Fi ───▶ Cloud ───▶ API ───▶ Mobile/Web App
```

| Layer | Main function |
|-------|---------------|
| **Device** | Identify rider (BLE + accelerometer) |
| **Edge** | EDGE Unit = BLE + GoPro control + motion detection + clip tagging + processing |
| **Cloud** | Storage, post‑processing, API |
| **App** | Registration, playback, user feedback |

---

## 2. Components
### 2.1 BearTag/TrickTag
- **Hardware:** BLE tracker with built‑in 3‑axis accelerometer, IP67, magnetic charger.
- **Advertising interval:** ≤ 100 ms (tuned later).
- **Payload:** `{tag_id, rssi, acc_x, acc_y, acc_z}`.
- **Battery:** Rechargeable Li‑Po; target ≥ 20 h active operation.

### 2.2 EDGE Unit
- **Recommended HW (MVP):** Raspberry Pi 4 + BLE dongle.
- **External camera:** GoPro HERO10+ (HindSight support).
- **GoPro communication:** Bidirectional API for preview stream + HindSight clip triggering.
- **Sensors:** BLE dongle, optional GPS module.
- **Pipeline:**
  1. Constant preview stream from GoPro via API.
  2. Motion Detection module runs continuously on preview stream.
  3. If motion is detected: trigger HindSight recording (15–30 s pre-roll).
  4. Clip is fetched via API → tagged with `tag_id`, GPS, UTC timestamp.
  5. Video sent to Process pipeline (ROI crop, virtual cameraman, compression).
  6. Async upload to cloud; clip removed after successful upload.
- **Note:** No local video recording is done on the EDGE device itself.

### 2.2.1 Motion Detection
- **Input:** Live preview stream from GoPro.
- **Output:** Trigger signal to GoPro API to start HindSight clip capture.
- **Function:** Constantly analyses motion in preview to detect potential tricks.
- **Trigger sensitivity:** Tunable – optimised to reduce false positives (e.g., ripples or background riders).

### 2.3 Cloud Platform
- **Storage:** Google Drive (account: `BearVisionApp@gmail.com`).
- **Post‑processing:** Stabilisation, slow‑motion, trick classification.
- **APIs:** REST & WebSocket push notifications.
- **Scalability:** Kubernetes / serverless functions.

### 2.4 Mobile/Web App
- **Features:** Tag registration, clip overview, sharing, mis‑match correction.
- **Tech:** React Native / PWA (decision pending).

---

## 3. Data flow & timing
1. **Tag approaches obstacle** → EDGE sees BLE RSSI > threshold.
1.5. Motion Detection runs on GoPro preview stream. If motion is detected, HindSight recording is triggered.
2. **Computer vision** detects rider in the action zone.
3. **Pre‑roll + event + post‑roll** are stored and cropped.  
4. Clip tagged with `tag_id`, GPS coords, UTC timestamp.  
5. Uploaded to cloud; push notification to app.

---

## 4. Offline robustness
- Ring buffer on EDGE covers 24 h of un‑uploaded clips.
- Retry queues with exponential back‑off.
- No clip is deleted until successfully uploaded or manually exported.

---

## 5. Privacy & GDPR
- ROI crop + optional blur outside ROI.
- On‑site signage ("Video recording by BearVision. Contact … to delete footage").
- Opt‑in consent in the app.

---

## 6. Hardware benchmarks (YOLOv8n @ 720p)
| Platform | FPS | Power | Price (DKK) |
|----------|-----|-------|-------------|
| RPi 4 (CPU only) | 2-3 | 5 W | 500 |
| RPi 5 + AI hat | 15-20 | 7 W | 800 |
| Jetson Orin Nano 8 GB | 40‑60 | 10 W | 1500 |

---

## 7. Open points / Roadmap
- Manual clip re‑assign in app (resolves ambiguity).  
- Multi‑obstacle coordination & central scheduler.  
- Fully automated labelling pipeline (CVAT export → YOLO import).  
- Battery test of BearTag/TrickTag in real sessions.

---

## 8. Glossary
| Term | Meaning |
|------|---------|
| **ROI** | Region of Interest – the area being cropped. |
| **Pre‑roll / Post‑roll** | Seconds of video kept before/after detection. |
| **RSSI** | Received Signal Strength Indicator. |
| **YOLO** | "You Only Look Once" – real‑time object detection.

---

> **Vision:** Deliver wake‑boarders instant access to their best tricks, captured and framed like a professional camera operator – completely automatically.

---
## Appendix A – Logical Block Overview

BearTag/TrickTag ─▶ BLE ─▶ EDGE Unit ─▶ GoPro HERO10+
                              │             ▲
                              ▼             │
                      Motion Detection ◀────┘
                              │
                              ▼
                         Process pipeline ──▶ Cloud

---
Denne version beskriver overgangen fra EDGE-optagelse til GoPro HindSight som primær metode for klipfangst. Lokal videooptagelse er droppet. Preview-stream anvendes til konstant bevægelsesanalyse via ny Motion Detection-blok.
