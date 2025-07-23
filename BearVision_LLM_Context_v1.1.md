# BearVision – LLM Context Doc v1.1
**Date:** 18‑07‑2025  
**Author:** Mr. Bear (Bjørn Eskildsen)

---

## Purpose
This document gives Large Language Models (LLMs) a concise, unambiguous domain overview of the BearVision project so the model can answer questions, propose improvements, and generate code/architectures consistently.

---

## 1. High‑level architecture
```
BearTag/TrickTag ───▶ BLE ───▶ EDGE Unit (camera + ML) ───▶ 4G/Wi‑Fi ───▶ Cloud ───▶ API ───▶ Mobile/Web App
```

| Layer | Main function |
|-------|---------------|
| **Device** | Identify rider (BLE + accelerometer) |
| **Edge** | Capture, crop, encode & buffer video |
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
- **Recommended HW (MVP):** Raspberry Pi 5 with the official AI hat.
- **Sensors:** 120° wide‑angle camera (≥ 1080p60), BLE dongle, GPS module.
- **ML pipeline:**
  1. 10 s ring buffer (pre‑roll).
  2. YOLOv8n rider detection → ROI crop & dynamic zoom ("Virtual Cameraman").
  3. Post‑roll (2 s) + trim.
  4. H.265 encode & write to local ring buffer (≥ 32 GB).
  5. Async upload with exponential back‑off; delete oldest clips when < 20 % free space.
- **User matching:** Session‑mode match on highest mean RSSI around event; ambiguous matches flagged to the app.

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
