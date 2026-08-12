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

The first version supports:

- simulation/hardware mode selection while idle;
- discovery of versioned YAML scenarios under `specs/scenarios`;
- wall-clock replay of behavioural scenario traces;
- live state and event log over Server-Sent Events;
- starting and stopping the Python runtime as a child process.

Scenario video and the real GoPro preview transport are deliberately not part
of this first slice. See `docs/remake/edge-control.md`.
