# GoPro Mock Usage

The integration tests rely on a lightweight GoPro mock to emulate
camera behaviour without hardware.

## Video files

- `test/input_video/TestMovie1.mp4` – high resolution sample used for
the streaming endpoint.
- `tests/data/preview_low.mp4` – low resolution version generated from
  the file above for preview algorithms.

To regenerate the preview file run:

```bash
python tools/make_preview_video.py test/input_video/TestMovie1.mp4 tests/data/preview_low.mp4
```

## Streaming interface

`tests.stubs.gopro.FakeGoPro` exposes a `streaming` component that
serves the high‑res video over HTTP when `start_stream` is invoked.
The returned URL points to a local HTTP server streaming the video
content in real time. Use `stop()` on the streaming object to shut the
server down when finished.
