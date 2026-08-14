# BearVision Server Control

Local administration UI for the authoritative Python server worker.

```powershell
cd apps/server-control
corepack pnpm install
corepack pnpm build
corepack pnpm serve
```

Open `http://127.0.0.1:4320`.

## Responsibilities

Python owns:

- paginated job and user read models;
- registry normalization and overlap validation;
- Box and filesystem job discovery;
- media download and SHA-256 verification;
- cached FFmpeg thumbnail generation;
- BearTag scoring and assignment decisions.

Node owns:

- the loopback-only HTTP server;
- mapping fixed routes to fixed Python CLI commands;
- static React assets;
- streaming Python-materialized media with HTTP Range support.

The UI provides an overview, a searchable video browser, job details with score
evidence, user and BearTag history, assignment preflight validation, and manual
requeue of unresolved or failed jobs.

Cached media is stored below the configured server `scratch_dir` in
`admin-media/<job-id>/`. The source file is accepted only after its size and
SHA-256 match the versioned job manifest.
