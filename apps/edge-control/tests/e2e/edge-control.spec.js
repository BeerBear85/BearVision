import { EventEmitter, once } from "node:events";
import { PassThrough } from "node:stream";

import { expect, test } from "@playwright/test";

import { createEdgeControlServer } from "../../server/server.mjs";

class FakeRuntime extends EventEmitter {
  constructor({ exitOnTerminate = false, runId } = {}) {
    super();
    this.pid = 4242;
    this.stdout = new PassThrough();
    this.stderr = new PassThrough();
    this.stdin = new PassThrough();
    this.exitOnTerminate = exitOnTerminate;
    this.runId = runId;
  }

  send(kind, payload, atSeconds = 0) {
    this.stdout.write(`${JSON.stringify({
      control_event_version: "1.1",
      run_id: this.runId,
      emitted_at: new Date().toISOString(),
      kind,
      at_s: atSeconds,
      payload,
    })}\n`);
  }

  kill(signal) {
    if (signal === "SIGKILL" || (signal === "SIGTERM" && this.exitOnTerminate)) {
      queueMicrotask(() => this.emit("exit", 0, signal));
    }
    return true;
  }
}

async function startFixture(options = {}) {
  const runtimes = [];
  const previousStopTimeout = process.env.BEARVISION_STOP_TIMEOUT_MS;
  if (options.stopTimeoutMs != null) {
    process.env.BEARVISION_STOP_TIMEOUT_MS = String(options.stopTimeoutMs);
  }
  let control;
  try {
    control = createEdgeControlServer({
      persistState: false,
      spawnRuntime: ({ runId }) => {
        const runtime = new FakeRuntime({ ...options.runtime, runId });
        runtimes.push(runtime);
        return runtime;
      },
      runReadiness: options.runReadiness,
    });
  } finally {
    if (previousStopTimeout == null) delete process.env.BEARVISION_STOP_TIMEOUT_MS;
    else process.env.BEARVISION_STOP_TIMEOUT_MS = previousStopTimeout;
  }
  control.server.listen(0, "127.0.0.1");
  await once(control.server, "listening");
  const { port } = control.server.address();
  return {
    url: `http://127.0.0.1:${port}`,
    runtimes,
    async close() {
      const closed = once(control.server, "close");
      control.close();
      await closed;
    },
  };
}

function readinessReport({ checks, warningIds = [] }) {
  return {
    readiness_schema_version: "1.0",
    checked_at: "2026-09-02T18:00:00Z",
    blocking: checks.some((check) => check.status === "fail" && check.critical),
    warning_ids: warningIds,
    checks,
  };
}

test("live monitoring stays active while background clip work progresses", async ({ page }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);

    await page.getByRole("button", { name: "Run scenario" }).click();
    const pipeline = page.getByRole("region", { name: "Pipeline" });
    await expect(pipeline.getByText("Monitoring", { exact: true }).first()).toBeVisible();

    const runtime = fixture.runtimes[0];
    runtime.send("capture_activity_changed", {
      activity: "capturing", request_id: "capture-7", pending_captures: 1,
    }, 1);
    await expect(page.getByRole("article", { name: "Live track" })).toContainText("Camera: Capturing");

    const clipJob = {
      job_id: "capture-7", request_id: "capture-7", processing_attempts: 1,
      queued_at_utc: "2026-09-03T10:00:00Z",
      state_changed_at_utc: "2026-09-03T10:00:01Z",
      raw_filename: "raw.mp4", processed_filename: null, failure_id: null,
    };
    runtime.send("clip_job_updated", {
      ...clipJob, status: "processing",
      counts: { queued: 0, processing: 1, failed: 0, completed: 0 },
    }, 2);
    await expect(page.getByRole("article", { name: "Background queue track" }).locator('[aria-current="step"]')).toContainText("Processing");
    await expect(pipeline).toContainText("Monitoring");

    runtime.send("clip_job_updated", {
      ...clipJob, status: "uploading",
      counts: { queued: 0, processing: 1, failed: 0, completed: 0 },
    }, 3);
    await expect(page.getByRole("article", { name: "Background queue track" }).locator('[aria-current="step"]')).toContainText("Uploading");

    runtime.emit("exit", 0, null);
    const recentRuns = page.getByRole("region", { name: "Recent runs" });
    await expect(recentRuns).toContainText("Completed");

    await page.reload();
    await expect(page.getByRole("region", { name: "Recent runs" })).toContainText("Completed");
    await expect(page.getByRole("button", { name: "Run scenario" })).toBeEnabled();
  } finally {
    await fixture.close();
  }
});

test("critical hardware failure blocks start and keeps corrective action visible", async ({ page }) => {
  const fixture = await startFixture({
    runReadiness: async () => readinessReport({
      checks: [{
        check_id: "camera",
        label: "GoPro camera",
        status: "fail",
        critical: true,
        evidence: "No preview frame arrived.",
        corrective_action: "Connect and power on the GoPro, then run readiness again.",
      }],
    }),
  });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Hardware" }).click();

    const readiness = page.getByRole("region", { name: "Hardware readiness" });
    await expect(readiness.getByRole("heading", { name: /Blocking issues/ })).toBeVisible();
    await expect(readiness).toContainText("No preview frame arrived.");
    await expect(readiness).toContainText("Connect and power on the GoPro, then run readiness again.");
    await expect(page.getByRole("button", { name: "Start hardware" })).toBeDisabled();

    await page.reload();
    await expect(page.getByRole("region", { name: "Hardware readiness" })).toContainText(
      "Connect and power on the GoPro, then run readiness again.",
    );
  } finally {
    await fixture.close();
  }
});

test("operator acknowledges a hardware warning by keyboard at 320 px and can start", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  const fixture = await startFixture({
    runReadiness: async () => readinessReport({
      warningIds: ["cloud_storage"],
      checks: [{
        check_id: "cloud_storage",
        label: "Upload storage",
        status: "warning",
        critical: false,
        evidence: "Cloud storage is temporarily unavailable.",
        corrective_action: "Continue locally or restore the network connection.",
      }],
    }),
  });
  try {
    await page.goto(fixture.url);
    const hardware = page.getByRole("button", { name: "Hardware" });
    await hardware.focus();
    await page.keyboard.press("Enter");

    const start = page.getByRole("button", { name: "Start hardware" });
    await expect(start).toBeDisabled();
    const acknowledgement = page.getByRole("checkbox", { name: "I reviewed this warning" });
    await acknowledgement.focus();
    await page.keyboard.press("Space");
    await expect(start).toBeEnabled();

    await start.focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("region", { name: "Pipeline" })).toContainText("Monitoring");
    await expect.poll(() => page.evaluate(
      () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
    )).toBe(true);
  } finally {
    await fixture.close();
  }
});

test("retryable failure survives refresh until the runtime resolves it", async ({ page }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    const runtime = fixture.runtimes[0];
    runtime.send("component_failed", {
      failure_id: "failure-publish-7",
      operation_id: "publish-7",
      stage: "uploading",
      component: "job_queue",
      error: "Queue write timed out.",
      operator_message: "The processing job could not be published.",
      corrective_action: "Check the local queue and retry the operation.",
      severity: "blocking",
      retryable: true,
      scope: "clip_job",
      job_id: "publish-7",
    });

    const failures = page.getByRole("region", { name: "Persistent failures" });
    await expect(failures).toContainText("The processing job could not be published.");
    await expect(failures.getByRole("button", { name: "Retry operation" })).toBeVisible();
    await expect(page.getByRole("region", { name: "Pipeline" })).toContainText("Monitoring");
    await expect(page.getByRole("button", { name: "Stop runtime" })).toBeVisible();

    await page.reload();
    const restoredFailures = page.getByRole("region", { name: "Persistent failures" });
    await expect(restoredFailures).toContainText("Check the local queue and retry the operation.");
    await restoredFailures.getByRole("button", { name: "Retry operation" }).click();
    await expect(restoredFailures).toContainText("The processing job could not be published.");

    runtime.send("failure_resolved", { failure_id: "failure-publish-7" });
    await expect(page.getByRole("region", { name: "Persistent failures" })).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Pipeline" })).toContainText("Monitoring");
  } finally {
    await fixture.close();
  }
});

test("terminal failure offers runtime restart but never operation retry", async ({ page }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    const runtime = fixture.runtimes[0];
    runtime.send("component_failed", {
      failure_id: "failure-camera-4",
      operation_id: "capture-4",
      stage: "recording",
      component: "camera",
      error: "Camera disconnected.",
      operator_message: "The camera connection was lost.",
      corrective_action: "Reconnect the camera, then restart the runtime.",
      severity: "terminal",
      retryable: false,
    });
    runtime.emit("exit", 1, null);

    const failures = page.getByRole("region", { name: "Persistent failures" });
    await expect(failures).toContainText("The camera connection was lost.");
    await expect(failures.getByRole("button", { name: "Retry operation" })).toHaveCount(0);
    const restart = page.getByRole("button", { name: "Restart runtime" });
    await expect(restart).toBeVisible();

    await restart.click();
    await expect(page.getByRole("region", { name: "Pipeline" })).toContainText("Monitoring");
    await expect(page.getByRole("region", { name: "Persistent failures" })).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Recent runs" })).toContainText("Failed");
    expect(fixture.runtimes).toHaveLength(2);
  } finally {
    await fixture.close();
  }
});

test("force stop appears only after graceful stop times out and requires confirmation", async ({ page }) => {
  const fixture = await startFixture({ stopTimeoutMs: 25 });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    await expect(page.getByRole("button", { name: "Force stop" })).toHaveCount(0);

    await page.getByRole("button", { name: "Stop runtime" }).click();
    const forceStop = page.getByRole("button", { name: "Force stop" });
    await expect(forceStop).toBeVisible();

    page.once("dialog", (dialog) => dialog.dismiss());
    await forceStop.click();
    await expect(forceStop).toBeFocused();

    page.once("dialog", async (dialog) => {
      expect(dialog.message()).toContain("incomplete artefacts");
      await dialog.accept();
    });
    await forceStop.click();

    await expect(page.getByRole("region", { name: "Recent runs" })).toContainText("Stopped");
  } finally {
    await fixture.close();
  }
});
