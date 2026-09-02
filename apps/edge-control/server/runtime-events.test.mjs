import assert from "node:assert/strict";
import test from "node:test";

import { parseRuntimeEventLine } from "./runtime-events.mjs";

test("runtime process events require the versioned envelope", () => {
  const event = parseRuntimeEventLine(JSON.stringify({
    control_event_version: "1.0",
    at_s: 1.5,
    kind: "person_detected",
    payload: { frame_id: "frame-1" },
  }));
  assert.equal(event.kind, "person_detected");
  assert.throws(
    () => parseRuntimeEventLine(JSON.stringify({ kind: "person_detected", payload: {} })),
    /version/,
  );
});
