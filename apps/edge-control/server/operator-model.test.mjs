import assert from "node:assert/strict";
import test from "node:test";

import {
  actionFailureNotice,
  deriveOperatorView,
  failurePresentation,
  mediaFailureNotice,
  pipelineForStage,
  reconcileActionNotice,
  restoreCapturedClip,
} from "../src/operator-model.js";

test("pipeline marks completed, current and upcoming stages", () => {
  const pipeline = pipelineForStage("packaging");
  assert.deepEqual(
    pipeline.map((item) => [item.key, item.status]),
    [
      ["readiness", "complete"],
      ["monitoring", "complete"],
      ["recording", "complete"],
      ["post_processing", "complete"],
      ["packaging", "current"],
      ["uploading", "upcoming"],
      ["complete", "upcoming"],
    ],
  );
});

test("operator view exposes only unresolved failures as requiring action", () => {
  const state = {
    mode: "hardware",
    active_run: {
      run_id: "run-1",
      stage: "failed",
      process_state: "running",
      stop_state: "none",
      failures: [
        { failure_id: "open", retryable: true, resolved_at: null },
        { failure_id: "closed", retryable: true, resolved_at: "2026-09-02T10:00:00Z" },
      ],
    },
    readiness: { blocking: false, warning_ids: ["ble"], checks: [] },
  };

  const view = deriveOperatorView(state, new Set(["ble"]));
  assert.deepEqual(view.unresolvedFailures.map((item) => item.failure_id), ["open"]);
  assert.equal(view.canStart, false);
  assert.equal(view.canStop, true);
  assert.equal(view.canForceStop, false);
  assert.equal(view.summary.code, "needs_attention");
});

test("simulation readiness and a finishing runtime have one operator status", () => {
  const ready = deriveOperatorView({
    mode: "simulation",
    active_run: null,
    readiness: null,
  });
  assert.deepEqual(
    [ready.summary.code, ready.summary.label, ready.summary.explanation],
    ["ready", "Simulation ready", "Physical equipment is not checked in simulation mode."],
  );

  const finishing = deriveOperatorView({
    mode: "simulation",
    active_run: {
      stage: "stopped",
      process_state: "running",
      stop_state: "none",
      failures: [],
    },
    readiness: null,
  });
  assert.equal(finishing.summary.code, "finishing");
  assert.equal(finishing.summary.headline, "Recording stopped — finishing the run");
});

test("a disconnected stop becomes an unconfirmed command with support context", () => {
  const notice = actionFailureNotice("stop", new TypeError("Failed to fetch"), {
    run: {
      run_id: "run-9",
      mode: "simulation",
      scenario: "single-rider-success.yaml",
      stage: "monitoring",
      process_state: "running",
    },
    streamConnected: false,
    lastKnownAt: "2026-09-09T08:00:00Z",
    occurredAt: "2026-09-09T08:00:05Z",
  });

  assert.equal(notice.status, "unconfirmed");
  assert.equal(notice.retryable, true);
  assert.equal(notice.runId, "run-9");
  assert.match(notice.headline, /could not confirm/);
  assert.match(notice.supportDetails, /Run: run-9/);
  assert.match(notice.supportDetails, /Last known runtime state: monitoring \/ running/);
});

test("an unconfirmed stop is cleared when a fresh snapshot has terminal recovery", () => {
  const notice = actionFailureNotice("stop", new TypeError("Failed to fetch"), {
    run: { run_id: "run-9", stage: "monitoring", process_state: "running" },
    streamConnected: false,
  });
  const state = {
    phase: "failed",
    active_run: { run_id: "run-9", stage: "failed", process_state: "exited" },
  };

  assert.equal(reconcileActionNotice(notice, state, true), null);
  assert.equal(reconcileActionNotice(notice, state, false), notice);
});

test("a missing tracking file is presented as a scoped media problem", () => {
  const notice = mediaFailureNotice("tracking", { code: "MEDIA_NOT_FOUND", message: "Capture does not exist." }, {
    run: { run_id: "run-4", stage: "completed", process_state: "exited" },
    filename: "clip.tracking.json",
  });

  assert.equal(notice.kind, "tracking");
  assert.match(notice.headline, /completed/i);
  assert.match(notice.message, /result remains Completed/i);
  assert.match(notice.supportDetails, /MEDIA_NOT_FOUND/);
});

test("camera failures explain impact while keeping the raw error separate", () => {
  const failure = {
    component: "camera",
    error: "injected camera capture failure",
    operator_message: "injected camera capture failure",
    corrective_action: "Review the technical details.",
  };
  const simulation = failurePresentation(failure, "simulation");
  const hardware = failurePresentation(failure, "hardware");

  assert.equal(simulation.headline, "The simulated camera could not record.");
  assert.match(simulation.impact, /No new test clips/);
  assert.match(hardware.correctiveAction, /camera power and connection/);
  assert.doesNotMatch(simulation.headline, /injected/);
});

test("control restart failures explain different simulation and hardware recovery", () => {
  const failure = {
    component: "control_server",
    error: "The control server restarted while the runtime was active.",
  };

  const simulation = failurePresentation(failure, "simulation");
  const hardware = failurePresentation(failure, "hardware");

  assert.match(simulation.correctiveAction, /End the failed run/);
  assert.match(hardware.impact, /cannot confirm whether/i);
  assert.match(hardware.correctiveAction, /Check the camera/i);
});

test("captured media can be restored from persisted run artefacts", () => {
  const clip = restoreCapturedClip({
    run_id: "run-1",
    scenario: "scenario-a.yaml",
    artefacts: [
      { kind: "capture", filename: "raw.mp4", size_bytes: 10, created_at: "2026-09-09T08:00:00Z" },
      { kind: "processed", filename: "upload.mp4", size_bytes: 8 },
      { kind: "debug", filename: "debug.mp4", size_bytes: 12 },
      { kind: "tracking", filename: "tracking.json" },
    ],
  });

  assert.equal(clip.url, "/api/captures/raw.mp4");
  assert.equal(clip.processed_url, "/api/captures/upload.mp4");
  assert.equal(clip.debug_url, "/api/captures/debug.mp4");
  assert.equal(clip.tracking_url, "/api/captures/tracking.json");
  assert.equal(clip.run_id, "run-1");
  assert.equal(clip.scenario, "scenario-a.yaml");
  assert.equal(clip.captured_at, "2026-09-09T08:00:00Z");
});
