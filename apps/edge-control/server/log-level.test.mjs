import assert from "node:assert/strict";
import test from "node:test";

import {
  appendRetainedTraceEvent,
  createRuntimeLogLevelClassifier,
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

test("open_gopro protocol dumps and all continuation lines are debug", () => {
  const classify = createRuntimeLogLevelClassifier();
  const protocolDump = [
    "2026-09-02 15:29:44,383 INFO open_gopro.gopro_base:",
    "-------------->>>>>>>>",
    '"id" : "Preview Stream",',
    '"protocol" : "Protocol.HTTP",',
    '"endpoint" : "gopro/camera/stream",',
    '"mode" : "start",',
    '"port" : "8554",',
    "<<<<<<<<--------------",
  ];

  assert.deepEqual(protocolDump.map(classify), protocolDump.map(() => "debug"));
});

test("application state changes remain info and warnings remain warnings", () => {
  const classify = createRuntimeLogLevelClassifier();

  assert.equal(classify("2026-09-02 15:30:00 INFO bearvision.edge.orchestrator: Hardware ready"), "info");
  assert.equal(classify("state = running"), "info");
  assert.equal(classify("2026-09-02 15:30:00 INFO open_gopro.features.streaming: Starting preview stream"), "info");
  assert.equal(classify("2026-09-02 15:30:01 WARNING open_gopro.gopro_base: Frame dropped"), "warning");
  assert.equal(classify("retrying stream"), "warning");
});

test("server-assigned levels override fallback message parsing", () => {
  assert.equal(runtimeLogLevel({
    kind: "runtime_log",
    payload: { level: "debug", message: "INFO open_gopro.gopro_base:" },
  }), "debug");
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
