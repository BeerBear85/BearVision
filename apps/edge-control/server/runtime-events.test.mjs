import assert from "node:assert/strict";
import test from "node:test";

import { parseRuntimeEventLine } from "./runtime-events.mjs";

test("runtime process events require the versioned envelope", () => {
  const event = parseRuntimeEventLine(JSON.stringify({
    control_event_version: "1.1",
    run_id: "run-edge-17",
    emitted_at: "2026-09-03T08:15:00Z",
    at_s: 1.5,
    kind: "person_detected",
    payload: { frame_id: "frame-1" },
  }));
  assert.equal(event.kind, "person_detected");
  assert.equal(event.run_id, "run-edge-17");
  assert.equal(event.emitted_at, "2026-09-03T08:15:00Z");
  assert.throws(
    () => parseRuntimeEventLine(JSON.stringify({ kind: "person_detected", payload: {} })),
    /version/,
  );
  assert.throws(
    () => parseRuntimeEventLine(JSON.stringify({
      control_event_version: "1.1", emitted_at: "2026-09-03T08:15:00Z",
      kind: "person_detected", payload: {},
    })),
    /run id/,
  );
  assert.throws(
    () => parseRuntimeEventLine(JSON.stringify({
      control_event_version: "1.1", run_id: "run-edge-17", emitted_at: "not-a-time",
      kind: "person_detected", payload: {},
    })),
    /emission time/,
  );
});
