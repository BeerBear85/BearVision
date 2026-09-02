# Edge Control physical readiness handshake plan

Status: implemented on 2026-09-02.

## Outcome

Replace OS-level device-presence checks with short, non-destructive handshakes
through the same GoPro preview and Bleak scanner implementations used by the
runtime. Hardware start remains blocked unless both handshakes succeed.

## Interfaces and test seams

- `check_edge_readiness(...) -> ReadinessReport` remains the operator-facing
  interface tested for blocking status, evidence and corrective action.
- `PhysicalReadinessHandshake` is the hardware seam. Its interface exposes only
  `check_camera_preview()` and `check_ble_scanner()`; the real adapter owns
  controller/scanner creation, timeouts and cleanup.
- Tests inject fake hardware dependencies through this seam. They verify
  observable outcomes and cleanup, not private call structure.

## Behaviour

### GoPro preview

1. Connect through the production asynchronous GoPro controller and camera
   adapter.
2. Start the UDP preview and open it through `OpenCvPreviewFrameSource`.
3. Require one valid frame within the configured timeout.
4. Always close the frame source, stop preview and disconnect, each with a
   bounded cleanup timeout.
5. Do not start recording, list media, download files or mutate stored media.

### BLE scanner

1. Create the production `BleakKBeaconSource`.
2. Start a real scan for a short configured interval.
3. Stop the scanner cleanly and report how many BearTag advertisements were
   observed. Zero tags is still ready: the handshake proves scanner access, not
   rider presence.
4. Bound the whole scan and cleanup path so a driver cannot hang preflight.

## Configuration

Add strict readiness settings with conservative defaults:

- GoPro preview handshake timeout;
- BLE scan duration;
- cleanup timeout.

## Verification and delivery

- Red/green tests for success, timeout/failure translation and cleanup.
- Existing readiness blocking/warning tests remain green.
- Full remake Python suite, Ruff, mypy, Edge Control Node tests and Vite build.
- Final standards/spec review, commit, and push to an isolated `codex/` branch.
