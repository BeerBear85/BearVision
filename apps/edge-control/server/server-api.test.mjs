import assert from "node:assert/strict";
import { EventEmitter, once } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";

import { createEdgeControlServer } from "./server.mjs";

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
