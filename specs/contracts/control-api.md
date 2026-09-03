# Edge Control API 2.0

Status: implemented.

The Node.js Edge Control server is a process supervisor, durable run-state
adapter and presentation backend. Python owns hardware readiness, capture,
processing and publication policy.

## Run state

`GET /api/health` returns the authoritative snapshot: selected mode, current
run, ten recent runs and the latest readiness report. An active run records its
lifecycle stage, current operation, process and stop state, failures, artefacts
and retained event evidence. Replacement runs carry `restart_of_run_id`; forced
stops report whether partial artefacts were retained. State is atomically
persisted below the configured scratch directory.

## HTTP endpoints

- `GET /api/scenarios` lists supported versioned scenario files.
- `POST /api/mode` selects `simulation` or `hardware` while idle.
- `GET /api/readiness` returns the latest hardware preflight report.
- `POST /api/readiness/run` executes Python-owned, bounded GoPro preview-frame
  and BLE scanner handshakes plus runtime, model, media, storage and upload checks.
- `POST /api/runs` validates first and then starts a run.
- `GET /api/runs/current`, `GET /api/runs?limit=10`, and
  `GET /api/runs/:runId` return durable run evidence.
- `POST /api/runs/:runId/stop` requests graceful shutdown.
- `POST /api/runs/:runId/force-stop` is accepted only after the stop timeout.
- `POST /api/runs/:runId/restart` replaces an exited failed run.
- `POST /api/runs/:runId/failures/:failureId/retry` retries only a
  Python-declared retryable operation.
- `GET /api/events` streams events with IDs, heartbeat, bounded replay and
  snapshot fallback.

Structured errors contain `code`, `error`, `corrective_action`, and `details`.
Critical readiness failures block hardware start. Each warning must be
explicitly acknowledged by ID in `acknowledged_warning_ids`.

## Python event and command boundary

Python emits a discriminated runtime-event envelope with
`control_event_version: "1.1"`, the Node-assigned `run_id`, a timezone-aware
Python `emitted_at`, a known `kind`, a typed payload and a non-negative scenario
`at_s` or `null`. Lifecycle and capture events use stable operation IDs. Failure
events carry operator text, corrective action, severity and retryability. Node
preserves Python timestamps for stages, failures, resolutions and artefacts and
rejects runtime events whose `run_id` does not match the active run.

Node sends versioned newline-delimited commands over standard input. Version
`1.0` supports `retry_failure`; publication is the only supported operation
retry because it retains idempotent context and does not repeat capture.

This API is intended for one nearby Edge node on a trusted local network.
Authentication and fleet management are not specified.
