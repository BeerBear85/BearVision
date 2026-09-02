import assert from "node:assert/strict";
import test from "node:test";

import { ReadinessService } from "./readiness-service.mjs";

function report({ blocking = false, warningIds = [] } = {}) {
  return {
    readiness_schema_version: "1.0",
    checked_at: "2026-09-02T10:00:00Z",
    blocking,
    warning_ids: warningIds,
    checks: [],
  };
}

test("critical readiness failures block hardware start", async () => {
  const service = new ReadinessService({ runCommand: async () => report({ blocking: true }) });
  await service.run();
  await assert.rejects(
    () => service.assertReady({ acknowledgedWarnings: [] }),
    (error) => error.code === "READINESS_BLOCKED" && error.status === 409,
  );
});

test("every readiness warning requires explicit acknowledgement", async () => {
  const service = new ReadinessService({
    runCommand: async () => report({ warningIds: ["ble", "disk_space"] }),
  });
  await service.run();

  await assert.rejects(
    () => service.assertReady({ acknowledgedWarnings: ["ble"] }),
    (error) => error.code === "READINESS_WARNING_ACKNOWLEDGEMENT_REQUIRED",
  );
  await assert.doesNotReject(
    () => service.assertReady({ acknowledgedWarnings: ["ble", "disk_space"] }),
  );
});

test("hardware start always reruns readiness", async () => {
  let calls = 0;
  const service = new ReadinessService({
    runCommand: async () => { calls += 1; return report(); },
  });
  await service.run();

  await service.assertReady({ acknowledgedWarnings: [] });

  assert.equal(calls, 2);
});
