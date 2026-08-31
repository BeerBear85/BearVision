# BearVision 3

BearVision creates hands-free wakeboard clips. Vision detects a person and
starts a fixed-duration GoPro clip. Edge publishes an anonymous Box job with the
reduced clip and all whole-clip BearTag observations. A Python server worker
selects identity from accelerometer activity, RSSI and the historical user
registry; vision does not identify the rider or determine a jump timestamp.

## Active architecture

The installable package lives in `src/bearvision` and uses ports for camera,
preview frames, BLE scanning, detection, storage and time. Behavioural scenarios
and the edge service execute the same `BearVisionOrchestrator`.

```text
GoPro preview -> FrameSource -> Detector -> BearVisionOrchestrator
BearTag BLE -------------------------------> |
GoPro capture <----------------------------- |
Box input queue <--------------------------- |

Box input queue -> Python server worker -> processed/user_<uuid>
                                      \----> unresolved / failed

Scenario events -> simulated adapters -> shared local queue -> same server worker
```

In production, Edge and server communicate only through Box. Edge Control
simulations use `temp/simulation-queue`, which the server can consume with
`config/server.local.yaml`. The active Android application,
Google Drive integration and previous physical simulation are outside the 3.0
runtime and have been moved to a separate local archive.

## Development

Python 3.12 and `uv` are the reference development environment on Windows and
64-bit Linux. The same runtime and configuration contracts apply on both.

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check src tests/remake tests/vision tests/end2end
uv run mypy
```

The CI coverage figure combines the fast behavioural suite with the heavier
real-video/YOLO suite. It measures both executed lines and decision branches;
CI rejects combined coverage below 85 %. Reproduce it from the development
environment:

```bash
uv run coverage erase
uv run coverage run --parallel-mode -m pytest
uv run coverage run --parallel-mode -m pytest tests/vision tests/end2end -q
uv run coverage combine
uv run coverage report
```

Run a versioned behavioural scenario:

```bash
uv run bearvision-simulate specs/scenarios/single-rider-success.yaml
```

Run a recorded-video input through the GoPro emulator and real YOLO detector:

```bash
uv sync --locked --extra dev
uv run python -m bearvision.control simulate specs/scenarios/wakeboard-video-yolo.yaml
```

Generate a scenario from a Blender scene export (MP4, one or more numbered
rider-motion JSON files and camera-info YAML in the same directory). Schema 2.0
exports provide each rider's `bear_tag_id`; later riders are aligned to the
video timeline using `timing.frame_start`:

```bash
uv run bearvision-generate-blender-scenario \
  tests/end2end/blender_scenes/wakeboard_fs360_60fps --force
```

The generated YAML is written under `specs/scenarios`, where Edge Control
discovers it automatically. Select Simulation, choose the generated Blender
scenario and click **Run scenario**.

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

Run the local server worker/admin GUI:

```bash
cd apps/server-control
corepack pnpm install
corepack pnpm build
corepack pnpm serve
```

To consume packages emitted by Edge Control simulation, set
`BEARVISION_SERVER_CONFIG=config/server.local.yaml` before starting Server
Control. Production continues to use `config/server.yaml` and Box.

Open `http://127.0.0.1:4320`. It binds only to loopback. The admin UI includes
a paginated video library, assignment evidence, queue operations and
user/BearTag history. Node remains a thin HTTP and process shell: Python owns
the read models, registry validation, Box downloads, checksum verification and
FFmpeg thumbnail generation. Node only streams cached media with HTTP byte
ranges so browser seeking works.

Install the runtime and start it directly on Windows or Linux:

```bash
uv sync --locked
uv run bearvision-edge --config config/edge.yaml
```

For a systemd-managed 64-bit Raspberry Pi OS deployment on a Raspberry Pi 4 or
5, run the Linux provisioning script from a complete checkout:

```bash
sudo bash scripts/setup-raspberry-pi.sh --device-id edge-pi-1
```

It deploys only the active runtime files and selected YOLO model, installs the
locked Python runtime, configures Raspberry Pi OS FFmpeg and
creates an unprivileged `bearvision-edge` systemd service. Existing files under
`/etc/bearvision` are preserved. Add Box credentials to
`/etc/bearvision/bearvision.env`, connect the GoPro and BLE adapter, and start
the service with `sudo systemctl start bearvision-edge`. Use `--start` to start
it immediately during provisioning.

The base runtime includes packaged FFmpeg and FFprobe binaries for local clip
extraction; no system-wide installation or administrator access is required.
Override them with `BEARVISION_FFMPEG` and `BEARVISION_FFPROBE` if needed.

Real operation additionally requires a supported GoPro, BLE hardware, valid Box
credentials and a populated server user registry.

## Offline YOLO fine-tuning and annotation

The `pretraining/` workspace is retained exclusively for preparing datasets,
annotating footage and fine-tuning YOLO offline. It is not imported or executed
by the Edge or server runtimes:

```bash
uv sync --extra training --extra gui
uv run python pretraining/annotation/annotation_gui_pyqt.py
uv run python pretraining/run_training_gui.py
```

Its configuration lives in `config/annotation-example.yaml`. Offline workflow
tests live in `tests/offline_yolo/`. Test videos remain under `test/` and
`tests/data/` and are stored using Git LFS.

## Legacy code

Unsupported pre-3.0 post-processing tools are preserved under
`legacy/post_processing/` for reference only. They are not installed, imported
by the active package, collected by the default test suite or covered by CI.
The supported virtual-cameraman implementation lives in
`src/bearvision/processing/virtual_cameraman.py`.

The previous Edge application, its modules and old hardware/unit tests are
similarly preserved under `legacy/edge_v2/`. Checked-in ONNX files remain under
`code/dnn_models/` as active model assets, not executable legacy code.

## Specifications

- Runtime and simulation: `docs/remake/foundation.md`
- Disk-backed GoPro simulator: `docs/remake/gopro-simulator.md`
- Edge computer and control GUI: `docs/remake/edge-control.md`
- Component behaviour: `specs/components/`
- Versioned scenarios: `specs/scenarios/`
- Architecture decisions: `specs/decisions/`

Product vision and unimplemented roadmap items are documented separately in
`BearVision_LLM_Context.md`; they must not be interpreted as implemented runtime
behaviour.
