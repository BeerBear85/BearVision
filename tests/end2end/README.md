# Real-video end-to-end tests

These tests exercise recorded video through the production YOLO adapter,
five-second FFmpeg extraction, Virtual Cameraman processing, the durable local
queue and the server-side BearTag assignment algorithm.

## Video observations and telemetry assumptions

RSSI and body-frame acceleration are not recoverable from pixels. The scenario
values below are deterministic estimates for system testing, not measured
ground truth. RSSI assumes a BLE reference near `-50 dBm` at one metre and an
outdoor line-of-sight distance of roughly 5-25 metres. Acceleration includes
Earth gravity and uses a plausible wakeboard active magnitude near 1.9-2.0 g.

| Video | Resolution / FPS | Visible riders | First YOLO detection | Active RSSI | Active acceleration (m/s²) |
| --- | --- | ---: | ---: | ---: | --- |
| `preview_low.mp4` | 320×180 / 29.97 | 1 | 6.01 s | -72 dBm | (4.2, 2.1, 18.8) |
| `TestMovie1.mp4` | 1920×1080 / 29.97 | 1 | 6.21 s | -72 dBm | (4.2, 2.1, 18.8) |
| `TestMovie3.avi` | 1352×760 / 29.00 | 1 | 3.31 s | -66 dBm | (3.8, 2.4, 18.5) |
| `TestMovie5_two_persons.mp4`, rider 1 | 2704×1520 / 60.00 | 1 of 2 | 2.40 s | -64 dBm | (4.8, 2.6, 19.2) |
| `TestMovie5_two_persons.mp4`, rider 2 | 2704×1520 / 60.00 | 2 of 2 | 12.00 s | -66 dBm | (4.4, 2.3, 18.9) |

`bear_tag_666` is used for the first or only rider. The second rider in the
two-person recording uses `bear_tag_667`; one physical BearTag ID cannot safely
identify two people in the same scenario.

Run the suite with:

```bash
uv run pytest tests/end2end/test_real_video_scenarios.py -v
```
