import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { RunState } from "./run-state.mjs";

function deterministicState(options = {}) {
  let id = 0;
  let tick = 0;
  return new RunState({
    now: () => new Date(Date.UTC(2026, 8, 2, 10, 0, tick++)).toISOString(),
    createId: (prefix) => `${prefix}-${++id}`,
    ...options,
  });
}

test("typed events build an authoritative run snapshot with persistent failures", () => {
  const state = deterministicState();
  const started = state.start({ mode: "hardware" });

  state.record({ kind: "lifecycle_changed", payload: {
    stage: "recording", operation_id: "capture-1",
  } });
  state.record({ kind: "component_failed", payload: {
    failure_id: "failure-upload-1",
    operation_id: "capture-1:publish",
    stage: "uploading",
    component: "job_queue",
    error: "Box is offline",
    operator_message: "The clip could not be uploaded.",
    corrective_action: "Check the network and Box connection, then retry.",
    severity: "blocking",
    retryable: true,
  } });

  const snapshot = state.snapshot();
  assert.equal(snapshot.active_run.run_id, started.run_id);
  assert.equal(snapshot.active_run.stage, "failed");
  assert.equal(snapshot.active_run.failures[0].failure_id, "failure-upload-1");
  assert.equal(snapshot.active_run.failures[0].retryable, true);
  assert.equal(snapshot.active_run.failures[0].resolved_at, null);
});

test("Python event identity and occurrence time remain authoritative", () => {
  const state = deterministicState();
  const started = state.start({ mode: "hardware" });
  const emittedAt = "2026-09-03T08:15:00.000Z";

  state.record({
    control_event_version: "1.1",
    run_id: started.run_id,
    emitted_at: emittedAt,
    at_s: null,
    kind: "component_failed",
    payload: {
      failure_id: "failure-camera-17",
      stage: "recording",
      component: "camera",
      error: "Camera disconnected",
      severity: "terminal",
      retryable: false,
    },
  });

  const run = state.snapshot().active_run;
  assert.equal(run.events[0].run_id, started.run_id);
  assert.equal(run.events[0].emitted_at, emittedAt);
  assert.equal(run.stage_started_at, emittedAt);
  assert.equal(run.failures[0].occurred_at, emittedAt);
  assert.throws(
    () => state.record({
      control_event_version: "1.1",
      run_id: "run-from-an-older-process",
      emitted_at: "2026-09-03T08:16:00.000Z",
      at_s: null,
      kind: "lifecycle_changed",
      payload: { stage: "uploading", operation_id: "stale-operation" },
    }),
    /does not match active run/,
  );
  assert.equal(state.snapshot().active_run.stage, "failed");
});

test("camera activity follows stable operation ids without leaving monitoring", () => {
  const state = deterministicState();
  state.start({ mode: "hardware" });
  state.record({ kind: "runtime_started", payload: { pid: 1234 } });
  state.record({ kind: "capture_started", payload: { operation_id: "capture-1" } });
  state.record({ kind: "capture_completed", payload: {
    operation_id: "capture-1", filename: "one.mp4", size_bytes: 10,
  } });
  state.record({ kind: "capture_started", payload: { operation_id: "capture-2" } });

  const run = state.snapshot().active_run;
  assert.equal(run.capture_activity.request_id, "capture-2");
  assert.equal(run.capture_activity.activity, "capturing");
  assert.equal(run.stage, "monitoring");
  assert.deepEqual(run.artefacts.map((item) => item.filename), ["one.mp4"]);
});

test("run state survives process restart and retains only ten recent runs", () => {
  const root = mkdtempSync(join(tmpdir(), "bearvision-control-state-"));
  const stateFile = join(root, "runs.json");
  const state = deterministicState({ stateFile });

  for (let index = 0; index < 12; index += 1) {
    state.start({ mode: "simulation", scenario: `scenario-${index}.yaml` });
    state.complete("completed");
  }

  const restored = deterministicState({ stateFile });
  const snapshot = restored.snapshot();
  assert.equal(snapshot.active_run, null);
  assert.equal(snapshot.recent_runs.length, 10);
  assert.equal(snapshot.recent_runs[0].scenario, "scenario-11.yaml");
});

test("an active process is recovered as an actionable failure after control restart", () => {
  const root = mkdtempSync(join(tmpdir(), "bearvision-control-recovery-"));
  const stateFile = join(root, "runs.json");
  const state = deterministicState({ stateFile });
  const started = state.start({ mode: "hardware" });
  state.record({ kind: "runtime_started", payload: { pid: 1234 } });

  const restored = deterministicState({ stateFile }).snapshot().active_run;

  assert.equal(restored.run_id, started.run_id);
  assert.equal(restored.stage, "failed");
  assert.equal(restored.process_state, "exited");
  assert.equal(restored.failures[0].failure_id, `failure-${started.run_id}-control-restart`);
  assert.match(restored.failures[0].corrective_action, /Restart the runtime/);
});

test("force stop is unavailable until the graceful timeout expires", () => {
  const state = deterministicState();
  state.start({ mode: "hardware" });
  state.requestStop();
  assert.equal(state.snapshot().active_run.stop_state, "graceful_requested");
  assert.throws(() => state.forceStop(), /not available/);

  state.allowForceStop();
  state.forceStop();
  assert.equal(state.snapshot().active_run.stop_state, "forced");
  assert.equal(state.snapshot().active_run.cleanup_status, "no_artefacts");
});

test("a resolved retryable failure remains in evidence with its resolution time", () => {
  const state = deterministicState();
  state.start({ mode: "hardware" });
  state.record({ kind: "component_failed", payload: {
    failure_id: "failure-upload",
    operation_id: "capture-1:publish",
    stage: "uploading",
    component: "job_queue",
    error: "offline",
    retryable: true,
  } });

  state.record({ kind: "failure_resolved", payload: {
    failure_id: "failure-upload",
    operation_id: "capture-1:publish",
  } });

  const failure = state.snapshot().active_run.failures[0];
  assert.notEqual(failure.resolved_at, null);
  assert.equal(failure.retryable, true);
});

test("an operator stop is retained as stopped rather than failed", () => {
  const state = deterministicState();
  state.start({ mode: "hardware" });

  const completed = state.complete("stopped");

  assert.equal(completed.stage, "stopped");
  assert.equal(state.snapshot().recent_runs[0].stage, "stopped");
});

test("clip queue events persist a bounded read model without changing monitoring", () => {
  const state = deterministicState();
  state.start({ mode: "hardware" });
  state.record({ kind: "runtime_started", payload: { pid: 123 } });
  const jobs = Array.from({ length: 25 }, (_, index) => ({
    job_id: `job-${index}`,
    request_id: `capture-${index}`,
    status: index === 0 ? "processing" : "completed",
    processing_attempts: 1,
    queued_at_utc: `2026-09-03T09:${String(index).padStart(2, "0")}:00Z`,
    state_changed_at_utc: `2026-09-03T10:${String(index).padStart(2, "0")}:00Z`,
    raw_filename: `raw-${index}.mp4`,
    processed_filename: null,
    failure_id: null,
  }));
  state.record({ kind: "clip_queue_snapshot", payload: {
    counts: { queued: 3, processing: 1, failed: 2, completed: 40 },
    current_job: "job-0",
    oldest_queued_at_utc: "2026-09-03T09:00:00Z",
    jobs,
  } });

  const run = state.snapshot().active_run;
  assert.equal(run.stage, "monitoring");
  assert.deepEqual(run.clip_queue.counts, {
    queued: 3, processing: 1, failed: 2, completed: 40,
  });
  assert.equal(run.clip_queue.jobs.length, 20);
  assert.equal(run.clip_queue.current_job, "job-0");
});

test("clip-job failure remains actionable without failing the runtime", () => {
  const state = deterministicState();
  state.start({ mode: "hardware" });
  state.record({ kind: "runtime_started", payload: { pid: 123 } });
  state.record({ kind: "capture_activity_changed", payload: {
    activity: "capturing", request_id: "capture-2", pending_captures: 1,
  } });
  state.record({ kind: "component_failed", payload: {
    failure_id: "failure-job-2-uploading-1",
    job_id: "job-2",
    scope: "clip_job",
    stage: "uploading",
    component: "job_queue",
    error: "Box offline",
    retryable: true,
  } });

  const run = state.snapshot().active_run;
  assert.equal(run.stage, "monitoring");
  assert.deepEqual(run.capture_activity, {
    activity: "capturing", request_id: "capture-2", pending_captures: 1,
  });
  assert.equal(run.failures[0].scope, "clip_job");
  assert.equal(run.failures[0].job_id, "job-2");
});

test("queue snapshot restores retryable clip-job failure cards", () => {
  const state = deterministicState();
  state.start({ mode: "hardware" });
  state.record({ kind: "runtime_started", payload: { pid: 123 } });
  state.record({ kind: "clip_queue_snapshot", payload: {
    counts: { queued: 0, processing: 0, failed: 1, completed: 4 },
    current_job: null,
    oldest_queued_at_utc: null,
    jobs: [{
      job_id: "job-failed",
      request_id: "capture-failed",
      status: "failed",
      processing_attempts: 1,
      queued_at_utc: "2026-09-03T09:00:00Z",
      state_changed_at_utc: "2026-09-03T09:01:00Z",
      raw_filename: "raw.mp4",
      processed_filename: null,
      failure_id: "failure-job-failed-uploading-1",
      failed_step: "uploading",
      technical_error: "Box offline",
    }],
  } });

  const run = state.snapshot().active_run;
  assert.equal(run.stage, "monitoring");
  assert.equal(run.failures[0].failure_id, "failure-job-failed-uploading-1");
  assert.equal(run.failures[0].scope, "clip_job");
  assert.equal(state.retryFailure(run.failures[0].failure_id).job_id, "job-failed");
});
