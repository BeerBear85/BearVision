import assert from "node:assert/strict";
import { EventEmitter, once } from "node:events";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { get } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import test from "node:test";

import { createEdgeControlServer, runJsonProcess } from "./server.mjs";

async function runningServer(options = {}) {
  const control = createEdgeControlServer(options);
  control.server.listen(0, "127.0.0.1");
  await once(control.server, "listening");
  const { port } = control.server.address();
  return {
    ...control,
    request: async (path, options) => {
      const response = await fetch(`http://127.0.0.1:${port}${path}`, options);
      return { response, body: await response.json() };
    },
  };
}

function fakeRuntimeChild() {
  const child = new EventEmitter();
  child.pid = 4321;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.stdin = new PassThrough();
  child.kill = () => true;
  return child;
}

test("invalid scenario is rejected before runtime state changes", async (context) => {
  const control = await runningServer({
    persistState: false,
    spawnRuntime: () => { throw new Error("must not spawn"); },
  });
  context.after(() => control.close());

  const { response, body } = await control.request("/api/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ mode: "simulation", scenario: "missing.yaml" }),
  });
  assert.equal(response.status, 400);
  assert.equal(body.code, "INVALID_REQUEST");

  const health = await control.request("/api/health");
  assert.equal(health.body.phase, "idle");
  assert.equal(health.body.active_run, null);
});

test("critical readiness produces a structured conflict and blocks hardware start", async (context) => {
  const readiness = {
    readiness_schema_version: "1.0",
    checked_at: "2026-09-02T10:00:00Z",
    blocking: true,
    warning_ids: [],
    checks: [{
      check_id: "camera",
      label: "GoPro camera",
      status: "fail",
      critical: true,
      evidence: "not connected",
      corrective_action: "Connect and power on the GoPro.",
    }],
  };
  const control = await runningServer({
    persistState: false,
    runReadiness: async () => readiness,
  });
  context.after(() => control.close());

  const checked = await control.request("/api/readiness/run", { method: "POST" });
  assert.equal(checked.response.status, 200);
  assert.equal(checked.body.blocking, true);

  const started = await control.request("/api/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ mode: "hardware", acknowledged_warning_ids: [] }),
  });
  assert.equal(started.response.status, 409);
  assert.equal(started.body.code, "READINESS_BLOCKED");
  assert.equal(started.body.corrective_action.includes("critical"), true);
});

test("high-volume debug output does not retain a full state snapshot per log line", async (context) => {
  const child = fakeRuntimeChild();
  const readiness = {
    assertReady: async () => {},
    current: () => null,
  };
  const control = createEdgeControlServer({
    persistState: false,
    readiness,
    spawnRuntime: () => child,
  });
  context.after(() => control.close());
  await control.supervisor.start({ mode: "hardware" });

  for (let index = 0; index < 200; index += 1) {
    child.stderr.write(`2026-09-03 DEBUG bleak.backends.bluezdbus.manager: poll ${index}\n`);
  }
  await new Promise((resolve) => setImmediate(resolve));

  const debugEvents = control.eventStream.history
    .map(({ event }) => event)
    .filter((event) => event.kind === "runtime_log");
  assert.equal(debugEvents.length, 200);
  assert.equal(
    debugEvents.every((event) => event.control_snapshot == null),
    true,
    "debug events must not amplify memory by retaining repeated control snapshots",
  );
  assert.ok(JSON.stringify(control.eventStream.history).length < 100_000);
});

test("preview response stays complete when a new frame replaces the measured file", async (context) => {
  const root = mkdtempSync(join(tmpdir(), "bearvision-preview-race-"));
  const previewPath = join(root, "live-preview.jpg");
  writeFileSync(previewPath, Buffer.alloc(96 * 1024, 1));
  const control = createEdgeControlServer({
    persistState: false,
    scratchRoot: root,
    readPreviewFrame: async (path) => {
      const nextFrame = Buffer.alloc(16 * 1024, 2);
      writeFileSync(path, nextFrame);
      return nextFrame;
    },
  });
  control.server.listen(0, "127.0.0.1");
  await once(control.server, "listening");
  const { port } = control.server.address();

  context.after(() => {
    control.close();
    control.server.closeAllConnections();
    rmSync(root, { recursive: true, force: true });
  });

  const outcome = await new Promise((resolve) => {
    const request = get({
      hostname: "127.0.0.1",
      port,
      path: "/api/preview/frame.jpg",
      headers: { connection: "close" },
      agent: false,
    }, (response) => {
      let receivedBytes = 0;
      response.on("data", (chunk) => { receivedBytes += chunk.length; });
      response.on("aborted", () => resolve({
        complete: false,
        declaredBytes: Number(response.headers["content-length"]),
        receivedBytes,
      }));
      response.on("end", () => resolve({
        complete: response.complete,
        declaredBytes: Number(response.headers["content-length"]),
        receivedBytes,
      }));
    });
    request.on("error", (error) => resolve({ complete: false, error: error.code }));
  });

  assert.equal(outcome.complete, true);
  assert.equal(outcome.declaredBytes, 16 * 1024);
  assert.equal(outcome.receivedBytes, 16 * 1024);
});

test("missing preview remains a structured service-unavailable response", async (context) => {
  const root = mkdtempSync(join(tmpdir(), "bearvision-preview-missing-"));
  const control = await runningServer({ persistState: false, scratchRoot: root });
  context.after(() => {
    control.close();
    rmSync(root, { recursive: true, force: true });
  });

  const { response, body } = await control.request("/api/preview/frame.jpg");

  assert.equal(response.status, 503);
  assert.equal(body.code, "PREVIEW_NOT_READY");
});

test("a terminal failed run can be ended without starting a replacement", async (context) => {
  const child = fakeRuntimeChild();
  let spawnCount = 0;
  const control = await runningServer({
    persistState: false,
    spawnRuntime: () => { spawnCount += 1; return child; },
  });
  context.after(() => control.close());

  const started = await control.request("/api/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ mode: "simulation", scenario: "single-rider-success.yaml" }),
  });
  child.emit("exit", 1, null);

  const ended = await control.request(`/api/runs/${started.body.active_run.run_id}/end`, {
    method: "POST",
  });

  assert.equal(ended.response.status, 200);
  assert.equal(ended.body.active_run, null);
  assert.equal(ended.body.recent_runs[0].stage, "failed");
  assert.equal(spawnCount, 1);
});

test("GoPro diagnostics exposes a validated layered report", async (context) => {
  const diagnostics = {
    diagnostics_schema_version: "1.0",
    checked_at: "2026-09-10T10:00:00Z",
    target: "172.24.106.51:8080",
    status: "fail",
    summary: "The GoPro is visible over USB, but its USB network connection is not ready.",
    checks: [{
      check_id: "usb_device",
      label: "USB device detection",
      status: "pass",
      evidence: "Detected GoPro.",
      corrective_action: null,
    }],
  };
  const control = await runningServer({
    persistState: false,
    runGoProDiagnostics: async () => diagnostics,
  });
  context.after(() => control.close());

  const { response, body } = await control.request("/api/readiness/diagnostics/gopro", {
    method: "POST",
  });

  assert.equal(response.status, 200);
  assert.deepEqual(body, diagnostics);
});

test("a timed-out JSON process is terminated and returns the configured error", async () => {
  const child = fakeRuntimeChild();
  let killedWith = null;
  child.kill = (signal) => {
    killedWith = signal;
    child.emit("exit", null, signal);
    return true;
  };

  const processResult = runJsonProcess("python", [], {
    spawnProcess: () => child,
    timeoutMs: 10,
    timeoutError: {
      code: "READINESS_TIMEOUT",
      message: "Readiness timed out.",
      correctiveAction: "Run advanced diagnostics.",
    },
  });
  const harnessTimeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error("Regression harness expired.")), 100);
  });

  await assert.rejects(
    () => Promise.race([processResult, harnessTimeout]),
    (error) => error.code === "READINESS_TIMEOUT" && error.status === 504,
  );
  assert.equal(killedWith, "SIGTERM");
});

test("start preflight publishes checking and keeps timeout authoritative", async (context) => {
  let calls = 0;
  let failStart;
  let markStartChecking;
  const startChecking = new Promise((resolve) => { markStartChecking = resolve; });
  const readyReport = {
    readiness_schema_version: "1.0",
    checked_at: "2026-09-02T10:00:00Z",
    blocking: false,
    warning_ids: [],
    checks: [{ check_id: "camera", status: "pass" }],
  };
  const timeout = Object.assign(new Error("Readiness timed out."), {
    code: "READINESS_TIMEOUT",
    status: 504,
    correctiveAction: "Run advanced diagnostics.",
  });
  const control = await runningServer({
    persistState: false,
    runReadiness: async () => {
      calls += 1;
      if (calls === 1) return readyReport;
      markStartChecking();
      return new Promise((_, reject) => {
        failStart = () => reject(timeout);
      });
    },
  });
  context.after(() => control.close());

  const initial = await control.request("/api/readiness/run", { method: "POST" });
  assert.equal(initial.body.status, "ready");

  const starting = control.request("/api/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ mode: "hardware", acknowledged_warning_ids: [] }),
  });
  await startChecking;

  const checking = await control.request("/api/health");
  assert.equal(checking.body.readiness.status, "checking");
  assert.deepEqual(checking.body.readiness.checks, []);

  failStart();
  const failedStart = await starting;
  assert.equal(failedStart.response.status, 504);
  assert.equal(failedStart.body.code, "READINESS_TIMEOUT");

  const [health, readiness] = await Promise.all([
    control.request("/api/health"),
    control.request("/api/readiness"),
  ]);
  for (const snapshot of [health.body.readiness, readiness.body]) {
    assert.equal(snapshot.status, "failed");
    assert.equal(snapshot.failure.code, "READINESS_TIMEOUT");
    assert.deepEqual(snapshot.checks, []);
  }

  const states = control.eventStream.history
    .map(({ event }) => event)
    .filter((event) => event.kind === "readiness_updated")
    .map((event) => event.payload.status);
  assert.deepEqual(states, ["checking", "ready", "checking", "failed"]);
});
