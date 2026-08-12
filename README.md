# BearVision 3

BearVision creates hands-free wakeboard clips. Vision detects a person and
starts a fixed-duration GoPro clip. Rider identity is selected afterwards from
whole-clip BearTag accelerometer activity and RSSI; vision does not identify the
rider or determine a jump timestamp.

## Active architecture

The installable package lives in `src/bearvision` and uses ports for camera,
preview frames, BLE scanning, detection, storage and time. Behavioural scenarios
and the edge service execute the same `BearVisionOrchestrator`.

```text
GoPro preview -> FrameSource -> Detector -> BearVisionOrchestrator
BearTag BLE -------------------------------> |
GoPro capture <----------------------------- |
Box storage <------------------------------- |

Scenario events -> simulated adapters -> same orchestrator
```

The current production storage provider is Box. The active Android application,
Google Drive integration and previous physical simulation are outside the 3.0
runtime and have been moved to a separate local archive.

## Development

Python 3.12 and `uv` are the reference development environment.

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check src tests/remake
uv run mypy
```

Run a versioned behavioural scenario:

```bash
uv run bearvision-simulate specs/scenarios/single-rider-success.yaml
```

Run the recorded-video scenario through the real YOLO detector:

```bash
uv sync --locked --extra edge --extra dev
uv run python -m bearvision.control simulate specs/scenarios/wakeboard-video-yolo.yaml
```

Run the Edge computer's Node.js/React control GUI:

```bash
cd apps/edge-control
corepack pnpm install
corepack pnpm build
corepack pnpm serve
```

Open `http://localhost:4310`. The GUI can select behavioural simulation or real
hardware, play an attached scenario video on the trace timeline and show its
component sources and events. After capture it can also play the frame-accurate
five-second clip produced locally by the Windows Edge media runtime. Physical
hardware preview proxying is not yet implemented.

Install edge dependencies and start the production service:

```bash
uv sync --locked --extra edge
uv run bearvision-edge --config config/edge.yaml
```

The Edge extra includes packaged FFmpeg and FFprobe binaries for local clip
extraction; no system-wide installation or administrator access is required.
Override them with `BEARVISION_FFMPEG` and `BEARVISION_FFPROBE` if needed.

Real operation additionally requires a supported GoPro, BLE hardware, valid Box
credentials and a populated `tag_registry` in `config/edge.yaml`.

## Training, annotation and post-processing

These remain part of version 3.0, but are separate workflows from the edge
runtime:

```bash
uv sync --extra training --extra gui
uv run python pretraining/annotation/annotation_gui_pyqt.py
uv run python run_train_gui.py
uv run python run_post_processing.py INPUT.mp4 OUTPUT.json
uv run python post_processing_gui.py
```

Their versioned configuration files live in `config/`. Test videos remain under
`test/` and `tests/data/` and are stored using Git LFS.

## Specifications

- Runtime and simulation: `docs/remake/foundation.md`
- Edge computer and control GUI: `docs/remake/edge-control.md`
- Component behaviour: `specs/components/`
- Versioned scenarios: `specs/scenarios/`
- Architecture decisions: `specs/decisions/`

Product vision and unimplemented roadmap items are documented separately in
`BearVision_LLM_Context.md`; they must not be interpreted as implemented runtime
behaviour.
