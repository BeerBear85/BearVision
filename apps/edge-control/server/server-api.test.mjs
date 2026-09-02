import assert from "node:assert/strict";
import { once } from "node:events";
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
