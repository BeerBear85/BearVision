import assert from "node:assert/strict";
import test from "node:test";

import {
  appendRetainedTraceEvent,
  runtimeLogLevel,
  showsAtMinimumLogLevel,
} from "../src/log-level.js";

function runtimeLog(message) {
  return { kind: "runtime_log", payload: { message } };
}

test("runtime log levels are inferred from their message", () => {
  assert.equal(runtimeLogLevel(runtimeLog("2026-09-02 DEBUG USB poll")), "debug");
  assert.equal(runtimeLogLevel(runtimeLog("INFO Camera ready")), "info");
  assert.equal(runtimeLogLevel(runtimeLog("WARNING Frame dropped")), "warning");
  assert.equal(runtimeLogLevel(runtimeLog("ERROR GoPro disconnected")), "error");
  assert.equal(runtimeLogLevel(runtimeLog("CRITICAL Runtime aborted")), "error");
  assert.equal(runtimeLogLevel(runtimeLog("Unstructured runtime output")), "info");
  assert.equal(runtimeLogLevel({ kind: "capture_completed" }), null);
});

test("minimum level filters runtime logs without hiding semantic events", () => {
  assert.equal(showsAtMinimumLogLevel(runtimeLog("DEBUG USB poll"), "info"), false);
  assert.equal(showsAtMinimumLogLevel(runtimeLog("WARNING Frame dropped"), "info"), true);
  assert.equal(showsAtMinimumLogLevel(runtimeLog("INFO Camera ready"), "warning"), false);
  assert.equal(showsAtMinimumLogLevel(runtimeLog("ERROR Camera failed"), "warning"), true);
  assert.equal(showsAtMinimumLogLevel({ kind: "capture_completed" }, "error"), true);
});

test("debug noise cannot evict more important trace history", () => {
  let events = [
    runtimeLog("ERROR Camera failed"),
    runtimeLog("WARNING Frame dropped"),
    { kind: "capture_completed" },
  ];

  for (let index = 0; index < 20; index += 1) {
    events = appendRetainedTraceEvent(events, runtimeLog(`DEBUG poll ${index}`), 2);
  }

  assert.equal(events.filter((event) => runtimeLogLevel(event) === "debug").length, 2);
  assert.equal(events.some((event) => runtimeLogLevel(event) === "warning"), true);
  assert.equal(events.some((event) => runtimeLogLevel(event) === "error"), true);
  assert.equal(events.some((event) => event.kind === "capture_completed"), true);
});
