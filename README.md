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



## Annotation Pipeline GUI

Run a minimal interface to preview YOLO detections and save the resulting dataset:

```bash
python pretraining/annotation/annotation_gui.py
```

Choose a video and output directory, then press **Run** to watch the video with green bounding boxes while frames and labels are exported.

The pipeline automatically splits rider trajectories when no detections are
seen for `detection_gap_timeout_s` seconds (default 3). Adjust this and other
options in `pretraining/annotation/sample_config.yaml`.

Tiny person detections can be ignored by tuning
`min_person_bbox_diagonal_ratio` (default `0.001`). The value represents the
minimum fraction of the image diagonal a person's bounding box must span to be
kept. Raise it to discard more distant subjects or lower it to retain smaller
ones.

## YOLOv8 Wakeboard Detector Training

Fine-tune a YOLOv8 model on a folder of images and YOLO-format labels:

```bash
python pretraining/train_yolo.py /path/to/dataset --epochs 100 --batch 8 --onnx-out wakeboard.onnx
```

Prerequisites:

- Install dependencies with `pip install -r requirements.txt`.
- The dataset directory must contain images and matching `.txt` annotation files using YOLO coordinates.

The script automatically creates train/val splits, fine-tunes the base model and exports the best weights as an ONNX file.
