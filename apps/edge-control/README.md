# BearVision Edge Control

The Edge Control application is the thin Node.js/React shell around the Python
BearVision 3 runtime. It does not contain detection, assignment, capture or
storage policy.

```powershell
cd apps/edge-control
corepack pnpm install
corepack pnpm build
corepack pnpm serve
```

Open `http://localhost:4310`.

The current version supports:

- simulation/hardware mode selection while idle;
- discovery of versioned YAML scenarios under `specs/scenarios`;
- wall-clock replay of behavioural scenario traces;
- a live pipeline, stage elapsed time and persistent failure cards;
- Python-owned hardware readiness with critical blocking checks and explicit
  warning acknowledgements;
- real, bounded GoPro preview-frame and BLE scanner handshakes before hardware
  start, with guaranteed camera cleanup and no recording or media mutation;
- durable active/recent run state restored after refresh or server restart;
- replayable live state over Server-Sent Events without per-event polling;
- guarded graceful stop, force-stop, whole-runtime restart and safe publication
  retry;
- live GoPro hardware preview as throttled JPEG snapshots;
- recorded scenario video, extracted clips, processed upload clips and tracking
  evidence when the selected scenario produces them;
- a responsive operator interface aligned with Server Control's visual system.

The hardware preview is intentionally operator-grade rather than full frame rate:
the Python runtime publishes four JPEG snapshots per second through the Node server.

Automatic component retries are disabled. Failures remain visible until Python
reports resolution or the operator restarts the exited runtime. Edge Control has
no authentication; expose port 4310 only on a trusted local network.

## Tests

```powershell
corepack pnpm test
corepack pnpm exec playwright install chromium
corepack pnpm test:e2e
```

The Playwright suite exercises the built application through its public HTTP and
SSE interfaces. On Windows it falls back to an installed Chrome when the bundled
Playwright Chromium is unavailable.
