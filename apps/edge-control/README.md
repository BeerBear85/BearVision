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
- live state and event log over Server-Sent Events;
- starting and stopping the Python runtime as a child process.
- live GoPro hardware preview as throttled JPEG snapshots;
- recorded scenario video, extracted clips, processed upload clips and tracking
  evidence when the selected scenario produces them;
- a responsive operator interface aligned with Server Control's visual system.

The hardware preview is intentionally operator-grade rather than full frame rate:
the Python runtime publishes four JPEG snapshots per second through the Node server.
