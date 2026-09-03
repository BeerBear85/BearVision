# Edge Control operator monitoring and recovery plan

Status: MVP and browser interaction coverage implemented on 2026-09-02;
remaining hardening is listed below.

## Problem

Edge Control exposes preview media, health indicators and a technical event
trace, but it does not present the Python runtime's real lifecycle or preserve
failure evidence across a browser refresh. Failures appear mainly as a generic
banner or log entries, while the production configuration can retry component
failures five times before the operator sees the final outcome.

The target user is an on-site operator controlling one nearby Edge node. The
product must make live operation and failures clear without turning the UI into
an engineering console.

## Goals

- Show the authoritative pipeline stage at a glance: readiness, monitoring,
  recording, post-processing, packaging, uploading, stopping and failure.
- Present every failure as a persistent, actionable record until it is
  resolved or superseded.
- Restore the active run, pipeline state, failures and recent evidence after a
  browser refresh or temporary SSE interruption.
- Allow only backend-declared safe retries, whole-runtime restart and guarded
  force-stop.
- Block hardware start on failed critical readiness checks while allowing an
  explicit override for warnings.

## Non-goals

- Fleet management or control of multiple Edge nodes.
- User, BearTag, rider-assignment or video-library administration.
- Automatic component retries in this version.
- Arbitrary restart controls for individual technical components.
- Exposing Edge Control outside a trusted local network.

## Required domain and API changes

### Authoritative run model

Introduce a versioned `RunSnapshot` owned by the Node control process and
derived only from typed Python events. It contains:

- `run_id`, mode, selected scenario and start/end timestamps;
- lifecycle stage and the time that stage began;
- connection and process-supervisor state;
- current operation, if any;
- unresolved failures;
- indexed artefacts produced by the run;
- stop state: `none`, `graceful_requested`, `force_available`, or `forced`.

The React application renders this snapshot. It must not infer lifecycle state
from log strings or scan the retained event list to determine whether capture
is active.

### Typed lifecycle events

Extend the Python control-event contract with typed lifecycle transitions for:

- `monitoring`;
- `recording`;
- `post_processing`;
- `packaging`;
- `uploading`;
- `stopping`;
- `failed`;
- `completed`.

Each transition carries `run_id`, `operation_id`, `stage`, `occurred_at` and
optional progress evidence. Capture-related events must carry a stable capture
or request identifier so repeated captures cannot be confused.

### Typed failures

Replace message-only component failures with a versioned failure payload:

- stable `failure_id` and `operation_id`;
- failed stage and component;
- operator-facing summary and corrective action;
- technical details for progressive disclosure;
- `severity`: warning, blocking or terminal;
- `retryable` and an explicit retry command identifier;
- timestamps and resolution state.

Python decides whether an operation is retryable. Node transports and stores
that decision; React never invents it.

### Readiness model

Add `GET /api/readiness` and `POST /api/readiness/run`. Checks return a stable
identifier, label, status (`pass`, `warning`, `fail`), criticality, evidence and
corrective action. Hardware start requires fresh results and rejects any
critical failure. The run request may include explicit acknowledgement IDs for
warnings; critical failures cannot be overridden.

Initial critical checks:

- GoPro connection and usable preview/capture path;
- BLE adapter and scanner availability;
- Python runtime, configuration and model loadability;
- FFmpeg/FFprobe availability;
- capture and scratch directories writable with sufficient free space;
- configured storage adapter credentials/connectivity when upload is enabled.

The checks belong behind a Python-owned preflight command. Node remains the
HTTP/process adapter and must not duplicate hardware or configuration policy.

### Commands

Add these endpoints with structured error responses and correct HTTP status
codes:

- `POST /api/runs` — validate first, then start transactionally;
- `POST /api/runs/:runId/stop` — request graceful shutdown;
- `POST /api/runs/:runId/force-stop` — available only after the stop timeout;
- `POST /api/runs/:runId/restart` — start a new run linked to the failed run;
- `POST /api/runs/:runId/failures/:failureId/retry` — accepted only when the
  Python runtime declares the operation retryable;
- `GET /api/runs/current` and `GET /api/runs?limit=10`;
- `GET /api/runs/:runId` for restored state and evidence.

Retrying a failed operation requires a real Python command seam and retained
operation context. It must not be implemented as a disguised full-runtime
restart. Start with operations that are already idempotent and resumable, such
as job publication; add other retryable operations only when their contract is
proven safe.

## Implementation sequence

### Phase 0 — Make failures truthful

1. Set `error_recovery.max_restarts` to `0` in the active Edge configuration.
2. Add a regression test proving one component failure produces one attempt
   and one visible failure event.
3. Document that automatic retry is deliberately disabled for this release.

Exit criterion: a transient component failure is visible immediately and is
not silently retried.

### Phase 1 — Correct process supervision

1. Refactor `server/server.mjs` into a small runtime supervisor module.
2. Validate scenario, configuration, paths and preflight results before
   transitioning from idle to starting.
3. Handle spawn error, exit, graceful stop and forced stop through one cleanup
   path.
4. Add `stopping`; after a configurable timeout expose force-stop, require
   confirmation in the UI, then report partial-artefact cleanup status.
5. Ensure failed start cannot leave the server permanently active without a
   child process.

Exit criterion: every process outcome returns the control server to a valid,
observable state.

### Phase 2 — Contract-first pipeline and failure state

1. Extend `src/bearvision/contracts/control_events.py` with lifecycle and
   failure payloads.
2. Emit lifecycle transitions from the orchestrator instead of relying on its
   private `state` field.
3. Replace the current coarse `ControlState` with the versioned `RunSnapshot`.
4. Fix repeated-capture tracking by using stable request/capture IDs.
5. Add structured API errors: code, operator message, corrective action and
   technical details.

Exit criterion: Node tests can reconstruct the same run snapshot from a
recorded event sequence without inspecting log text.

### Phase 3 — Hardware readiness

1. Implement the Python-owned preflight command and typed result contract.
2. Replace the Linux-only GoPro sysfs check with adapter-aware checks that work
   on supported Windows and Linux deployments.
3. Run readiness automatically when Hardware mode is selected and revalidate
   immediately before start.
4. Add a readiness panel grouped into Passed, Warnings and Blocking issues.
5. Require explicit acknowledgement for warnings and reject starts with
   critical failures.

Exit criterion: disconnecting each critical dependency blocks start and shows
one concrete corrective action.

### Phase 4 — Durable state and reliable live transport

1. Persist the current and last ten run snapshots as atomic JSON files below a
   dedicated Edge Control state directory.
2. Add event IDs, heartbeat messages and a bounded replay buffer to SSE.
3. Support `Last-Event-ID`; fall back to a complete current snapshot when the
   replay window has expired.
4. Remove the React health request currently issued after every SSE event.
5. Restore current run, failures, stage and artefacts during page load.

Exit criterion: refresh or disconnect/reconnect does not lose the current stage
or unresolved failure evidence.

### Phase 5 — Operator workspace redesign

1. Make preview and the pipeline stage the dominant workspace.
2. Add a stage timeline with current activity, stage elapsed time and status
   using text and shape as well as colour.
3. Replace the dismissible generic error banner with persistent failure cards.
4. Each card shows the plain-language cause and corrective action; technical
   evidence is collapsed by default.
5. Move the raw event trace into a collapsible Diagnostics section.
6. Show Retry only for backend-declared retryable failures.
7. Show Restart runtime for a failed run. Reveal Force-stop only after the
   graceful-stop timeout and require confirmation.
8. Add a compact recent-run list and post-run verification summary.

Exit criterion: an operator can identify the current stage and the next safe
action without reading the event trace.

### Phase 6 — Safe operation retry

1. Add a versioned command channel between Node and the Python control process.
2. Retain the minimum operation context required to resume explicitly
   supported operations.
3. Implement job-publication retry first because queue publication is already
   designed to be idempotent.
4. Resolve the original failure only after a successful retry event; retain the
   complete attempt history.
5. Reject stale, unknown, non-retryable or already-resolved failure IDs.

Exit criterion: repeated Retry requests produce at most one committed result
and never duplicate a capture or published job.

## Test plan

### Python contract and runtime tests

- Every lifecycle transition and failure payload validates strictly.
- Unknown fields, invalid stages and missing operation IDs are rejected.
- With automatic retry disabled, component failure emits once and remains
  visible.
- Retryable publication can resume idempotently; capture/processing failures
  remain non-retryable until explicitly supported.
- Preflight results are correct on Windows and Linux adapter boundaries.

### Node supervisor and API tests

- Validation failure leaves the runtime idle.
- Spawn error, non-zero exit, graceful stop timeout and force-stop each produce
  the expected snapshot.
- Critical readiness failures block start; acknowledged warnings do not.
- SSE replay, heartbeat and snapshot fallback preserve event ordering.
- Atomic state restoration tolerates a missing or corrupt previous snapshot.
- Retry and force-stop endpoints enforce their state and identity guards.

### Browser interaction tests

- Complete configure → preflight → start → monitor → stop/complete workflow.
- Recording, processing and uploading stages are announced accessibly.
- Persistent failure survives refresh and exposes corrective action.
- Retry is absent for non-retryable failures.
- Force-stop remains hidden until the timeout and requires confirmation.
- Keyboard navigation, focus restoration and 320 px layout remain usable.

Use interaction tests rather than the current source-string assertions for
behavioural coverage.

## Remaining hardening

- Decide a deployment-specific cleanup policy for force-stopped partial files.
  The API currently reports whether partial artefacts were retained and avoids
  deleting operator evidence automatically.

## Completed hardening

- Added Playwright coverage for the complete simulation lifecycle, critical
  readiness blocking, explicit warning acknowledgement, persistent failures,
  safe retry, restart, delayed force-stop confirmation, refresh restoration,
  keyboard operation and the 320 px layout.
- Prevented stale HTTP command responses from replacing newer authoritative SSE
  snapshots by comparing control-event sequence numbers in the browser.
- Added Python-owned `run_id` and timezone-aware `emitted_at` metadata to the
  runtime event envelope. Node now preserves occurrence times and rejects stale
  events from a different run instead of relabelling them.

## Delivery slices and dependencies

```text
Phase 0 truthfulness
  -> Phase 1 supervision
  -> Phase 2 typed state
      -> Phase 3 readiness
      -> Phase 4 persistence/SSE
          -> Phase 5 operator UI
          -> Phase 6 safe retry
```

Phases 0–2 are the minimum coherent foundation. Phases 3–5 deliver the main
operator value. Phase 6 is separate because safe operation retry crosses the
Node/Python process boundary and must not be faked in the UI.

## Acceptance outcomes

- The operator sees a critical hardware failure before Start and receives a
  concrete corrective action.
- During a run, the UI shows the authoritative pipeline stage within one event
  update and never derives it from log messages.
- A failure remains visible across refresh until retry, restart or a later run
  resolves or supersedes it.
- No automatic retry occurs in this release.
- Retry is offered only for an operation proven idempotent by the backend.
- Graceful Stop is the normal action; Force-stop appears only after timeout.
- Reconnecting the browser preserves the current run and its evidence.

## Success measures

- 100% of injected critical failures display the correct failed stage and a
  corrective action in end-to-end tests.
- 0 silent automatic retries in the deployed Edge configuration.
- 0 lost active-run snapshots across tested browser refresh and SSE reconnect
  scenarios.
- An operator can identify stage and next action in under ten seconds during a
  supervised usability check.
