import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveOperatorView,
  pipelineForStage,
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
});

test("captured media can be restored from persisted run artefacts", () => {
  const clip = restoreCapturedClip({
    artefacts: [
      { kind: "capture", filename: "raw.mp4", size_bytes: 10 },
      { kind: "processed", filename: "upload.mp4", size_bytes: 8 },
      { kind: "debug", filename: "debug.mp4", size_bytes: 12 },
      { kind: "tracking", filename: "tracking.json" },
    ],
  });

  assert.equal(clip.url, "/api/captures/raw.mp4");
  assert.equal(clip.processed_url, "/api/captures/upload.mp4");
  assert.equal(clip.debug_url, "/api/captures/debug.mp4");
  assert.equal(clip.tracking_url, "/api/captures/tracking.json");
});
