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
      runGoProDiagnostics: options.runGoProDiagnostics,
    });
    if (!options.useStartupDefault) control.state.selectMode(options.initialMode ?? "simulation");
    if (options.initialRun) control.state.start(options.initialRun);
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

test("hardware is the safe startup default", async ({ page }) => {
  const fixture = await startFixture({ useStartupDefault: true });
  try {
    await page.goto(fixture.url);

    await expect(page.getByRole("button", { name: "Hardware", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("heading", { name: "Check the hardware before starting" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Run readiness", exact: true }).first()).toBeEnabled();
    await expect(page.getByRole("button", { name: "Start hardware" })).toHaveCount(0);
    await expect(page.getByRole("combobox", { name: "Scenario" })).toHaveCount(0);
  } finally {
    await fixture.close();
  }
});

test("readiness details and logs live on separate routes", async ({ page }) => {
  const fixture = await startFixture({ useStartupDefault: true });
  try {
    await page.goto(fixture.url);

    await expect(page.getByRole("region", { name: "Readiness summary" })).toBeVisible();
    await expect(page.getByRole("region", { name: "Hardware readiness" })).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Detailed logs" })).toHaveCount(0);

    await page.getByRole("link", { name: "Readiness" }).click();
    await expect(page).toHaveURL(/\/readiness$/);
    await expect(page.getByRole("region", { name: "Hardware readiness" })).toBeVisible();

    await page.getByRole("link", { name: "Logs" }).click();
    await expect(page).toHaveURL(/\/logs$/);
    await expect(page.getByRole("region", { name: "Detailed logs" })).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Minimum log level" })).toBeVisible();
  } finally {
    await fixture.close();
  }
});

test("a repeated readiness check hides the previous failure until the new result arrives", async ({ page }) => {
  let callCount = 0;
  let finishSecondCheck;
  const failedReport = readinessReport({
    checks: [{
      check_id: "camera",
      label: "GoPro camera",
      status: "fail",
      critical: true,
      evidence: "Previous camera failure.",
      corrective_action: "Reconnect the camera.",
    }],
  });
  const fixture = await startFixture({
    runReadiness: async () => {
      callCount += 1;
      if (callCount === 1) return failedReport;
      return new Promise((resolve) => { finishSecondCheck = () => resolve(failedReport); });
    },
  });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Hardware" }).click();
    await page.getByRole("link", { name: /Readiness/ }).click();
    await expect(page.getByText("Previous camera failure.")).toBeVisible();

    await page.getByRole("button", { name: "Run readiness" }).click();
    await expect(page.getByText(/Previous results are hidden until this check finishes/)).toBeVisible();
    await expect(page.getByText("Previous camera failure.")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Checking…" })).toBeDisabled();

    finishSecondCheck();
    await expect(page.getByText("Previous camera failure.")).toBeVisible();
  } finally {
    await fixture.close();
  }
});

test("the default simulation uses the repository input test video", async ({ page }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);

    await expect(page.getByRole("combobox", { name: "Scenario" })).toHaveValue(
      "wakeboard-testmovie1-yolo.yaml",
    );
    await page.getByRole("button", { name: "Run scenario" }).click();
    await expect(page.getByText(/input test video/)).toBeVisible();
    await expect(page.getByText(/first run can take up to a minute/i)).toBeVisible();
    const monitoringStep = page
      .getByRole("article", { name: "Live track" })
      .locator("li")
      .filter({ hasText: "Monitoring" });
    await expect(monitoringStep).toHaveClass("upcoming");
    await expect(monitoringStep).not.toHaveAttribute("aria-current", "step");
  } finally {
    await fixture.close();
  }
});

test("hardware startup guidance describes live equipment instead of simulation", async ({ page }) => {
  const fixture = await startFixture({
    initialMode: "hardware",
    initialRun: { mode: "hardware" },
  });
  try {
    await page.goto(fixture.url);

    await expect(page.getByRole("heading", { name: "BearVision is starting" })).toBeVisible();
    await expect(page.getByText(/Connecting to the GoPro and starting BearTag monitoring/)).toBeVisible();
    await expect(page.getByText(/input test video|replay starts/i)).toHaveCount(0);
  } finally {
    await fixture.close();
  }
});

test("operator timestamps use Danish 24-hour formatting", async ({ page }) => {
  const fixture = await startFixture({
    runReadiness: async () => readinessReport({ checks: [] }),
  });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Hardware" }).click();

    const summary = page.getByRole("region", { name: "Readiness summary" });
    await expect(summary).toContainText("Ready");
    const checkedAt = await summary.locator("small").innerText();
    expect(checkedAt).toMatch(/\b\d{2}\.\d{2}\b/);
    expect(checkedAt).not.toMatch(/\b(?:AM|PM)\b/);
  } finally {
    await fixture.close();
  }
});

test("a completed stop explains retained outputs and remaining clip work", async ({ page }) => {
  const fixture = await startFixture({ runtime: { exitOnTerminate: true } });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    fixture.runtimes[0].send("capture_completed", {
      asset_id: "capture-1",
      filename: "capture-1.mp4",
      size_bytes: 2048,
      clip_start_s: 0,
      clip_duration_s: 4,
    });

    await page.getByRole("button", { name: "Stop runtime" }).click();

    const outcome = page.getByRole("status");
    await expect(outcome).toContainText("BearVision stopped");
    await expect(outcome).toContainText("1 output file retained");
    await expect(outcome).toContainText("No clips were active or queued");
  } finally {
    await fixture.close();
  }
});

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
    await page.getByRole("button", { name: "Review blocking issues" }).click();

    const readiness = page.getByRole("region", { name: "Hardware readiness" });
    await expect(readiness.getByRole("heading", { name: /Blocking issues/ })).toBeVisible();
    await expect(readiness).toContainText("No preview frame arrived.");
    await expect(readiness).toContainText("Connect and power on the GoPro, then run readiness again.");
    await expect(page.getByRole("button", { name: "Start hardware" })).toHaveCount(0);

    await page.reload();
    await expect(page.getByRole("region", { name: "Hardware readiness" })).toContainText(
      "Connect and power on the GoPro, then run readiness again.",
    );
  } finally {
    await fixture.close();
  }
});

test("failed GoPro readiness offers layered advanced diagnostics", async ({ page }) => {
  const fixture = await startFixture({
    runReadiness: async () => readinessReport({
      checks: [{
        check_id: "camera",
        label: "GoPro camera",
        status: "fail",
        critical: true,
        evidence: "No preview frame arrived.",
        corrective_action: "Reconnect the camera.",
      }],
    }),
    runGoProDiagnostics: async () => ({
      diagnostics_schema_version: "1.0",
      checked_at: "2026-09-10T10:00:00Z",
      target: "172.24.106.51:8080",
      status: "fail",
      summary: "The GoPro is visible over USB, but its USB network connection is not ready.",
      checks: [
        { check_id: "usb_device", label: "USB device detection", status: "pass", evidence: "Detected GoPro HERO.", corrective_action: null },
        { check_id: "usb_network", label: "USB network interface", status: "fail", evidence: "No IPv4 interface can reach the camera subnet.", corrective_action: "Reconnect the cable." },
        { check_id: "camera_tcp", label: "Camera API connection", status: "fail", evidence: "TCP connection failed.", corrective_action: "Check the interface." },
        { check_id: "camera_http", label: "GoPro HTTP communication", status: "fail", evidence: "No valid response.", corrective_action: "Restart the GoPro." },
      ],
    }),
  });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Hardware" }).click();
    await page.getByRole("button", { name: "Review blocking issues" }).click();

    const launch = page.getByRole("button", { name: "Run advanced diagnostics" });
    await expect(launch).toBeVisible();
    await launch.click();

    const diagnostics = page.getByRole("region", { name: "GoPro connection path" });
    await expect(diagnostics).toContainText("Detected GoPro HERO.");
    await expect(diagnostics).toContainText("No IPv4 interface can reach the camera subnet.");
    await expect(diagnostics).toContainText("do not start preview or recording");
  } finally {
    await fixture.close();
  }
});

test("readiness timeout opens a safe path to GoPro diagnostics", async ({ page }) => {
  let callCount = 0;
  const fixture = await startFixture({
    runReadiness: async () => {
      callCount += 1;
      if (callCount === 1) return readinessReport({ checks: [] });
      const error = new Error("Hardware readiness did not finish within 75 seconds.");
      Object.assign(error, {
        code: "READINESS_TIMEOUT",
        status: 504,
        correctiveAction: "Run advanced GoPro diagnostics, then retry readiness.",
      });
      throw error;
    },
    runGoProDiagnostics: async () => ({
      diagnostics_schema_version: "1.0",
      checked_at: "2026-09-10T10:00:00Z",
      target: "172.24.106.51:8080",
      status: "pass",
      summary: "The Edge computer can communicate with the GoPro HTTP API.",
      checks: [{
        check_id: "usb_device",
        label: "USB device detection",
        status: "pass",
        evidence: "Detected GoPro HERO12 Black.",
        corrective_action: null,
      }],
    }),
  });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Hardware" }).click();
    await page.getByRole("link", { name: /Readiness/ }).click();
    await page.getByRole("button", { name: "Run readiness" }).click();

    await expect(page).toHaveURL(/\/readiness$/);
    await expect(page.getByRole("heading", { name: "Readiness timed out" })).toBeVisible();
    await page.reload();
    await expect(page.getByRole("heading", { name: "Readiness timed out" })).toBeVisible();
    const launch = page.getByRole("button", { name: "Run advanced GoPro diagnostics" });
    await expect(launch).toBeVisible();
    await launch.click();

    const diagnostics = page.getByRole("region", { name: "GoPro connection path" });
    await expect(diagnostics).toContainText("Detected GoPro HERO12 Black.");
    await expect(diagnostics).toContainText("communicate with the GoPro HTTP API");
    await expect(page.getByRole("button", { name: "Run readiness" })).toBeEnabled();
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

    const reviewWarning = page.getByRole("button", { name: "Review readiness warning" });
    await expect(reviewWarning).toBeVisible();
    await reviewWarning.focus();
    await page.keyboard.press("Enter");
    const acknowledgement = page.getByRole("checkbox", { name: "I reviewed this warning" });
    await acknowledgement.focus();
    await page.keyboard.press("Space");
    await page.getByRole("link", { name: "Overview" }).click();
    const start = page.getByRole("button", { name: "Start hardware" });
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
    await expect(failures).toContainText("The simulated camera could not record.");
    await expect(failures).toContainText("No new test clips are being captured.");
    await expect(failures.getByText("Camera disconnected.", { exact: true })).toBeHidden();
    await failures.getByText("Technical details").click();
    await expect(failures.getByText("Camera disconnected.", { exact: true })).toBeVisible();
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

test("a terminal failed simulation can be ended without restarting it", async ({ page }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);
    const scenario = page.getByRole("combobox", { name: "Scenario" });
    await page.getByRole("button", { name: "Run scenario" }).click();
    const runtime = fixture.runtimes[0];
    runtime.send("component_failed", {
      failure_id: "failure-camera-end",
      component: "camera",
      error: "Camera disconnected.",
      severity: "terminal",
      retryable: false,
    });
    runtime.emit("exit", 1, null);

    await page.getByRole("button", { name: "End failed run" }).click();

    await expect(scenario).toBeEnabled();
    await expect(page.getByRole("button", { name: "Run scenario" })).toBeEnabled();
    await expect(page.getByRole("region", { name: "Recent runs" })).toContainText("Failed");
    expect(fixture.runtimes).toHaveLength(1);
  } finally {
    await fixture.close();
  }
});

test("a missing tracking file stays scoped to the completed result", async ({ page }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    const runtime = fixture.runtimes[0];
    runtime.send("capture_completed", {
      filename: "missing-result-source.mp4",
      size_bytes: 2048,
    });
    runtime.send("virtual_cameraman_completed", {
      processed_filename: "missing-result-processed.mp4",
      tracking_filename: "missing-result.tracking.json",
      debug_video_filename: "missing-result-debug.mp4",
      processed_size_bytes: 1024,
    });
    runtime.emit("exit", 0, null);

    await expect(page.getByRole("region", { name: "Recent runs" })).toContainText("Completed");
    await expect(page.getByText("Run completed — tracking view unavailable")).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Try tracking again" })).toBeVisible();
  } finally {
    await fixture.close();
  }
});

test("changing scenario clears media and overlay from the previous run", async ({ page }) => {
  const fixture = await startFixture({ runtime: { exitOnTerminate: true } });
  try {
    await page.goto(fixture.url);
    const scenario = page.getByRole("combobox", { name: "Scenario" });
    const initialScenario = await scenario.inputValue();
    const otherScenario = await scenario.locator("option").evaluateAll(
      (options, current) => options.map((option) => option.value).find((value) => value !== current),
      initialScenario,
    );
    expect(otherScenario).toBeTruthy();

    await page.getByRole("button", { name: "Run scenario" }).click();
    const runtime = fixture.runtimes[0];
    runtime.send("person_detected", {
      bounding_box: { x_px: 1, y_px: 1, width_px: 10, height_px: 10 },
      confidence: 0.9,
      coordinate_space: { width_px: 100, height_px: 100 },
    }, 2);
    runtime.send("capture_completed", { filename: "old-run.mp4", size_bytes: 1024 }, 3);
    await expect(page.getByRole("button", { name: "Extracted clip" })).toBeVisible();
    await expect(page.getByText("T+ 3.0 s", { exact: true })).toBeVisible();

    await page.getByRole("button", { name: "Stop runtime" }).click();
    await expect(scenario).toBeEnabled();
    await scenario.selectOption(otherScenario);

    await expect(page.getByRole("button", { name: "Extracted clip" })).toHaveCount(0);
    await expect(page.getByText("T+ 3.0 s", { exact: true })).toHaveCount(0);
    await expect(page.locator(".tracking-overlay")).toHaveCount(0);
  } finally {
    await fixture.close();
  }
});

test("offline stop is shown as unconfirmed and can be retried", async ({ page, context }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    await expect(page.getByRole("heading", { name: "BearVision is starting" })).toBeVisible();

    await context.setOffline(true);
    await page.getByRole("button", { name: "Stop runtime" }).click();
    const notice = page.getByRole("alert");
    await expect(notice).toContainText("We could not confirm that BearVision stopped");
    await expect(notice).toContainText("BearVision may still be recording");
    await expect(notice.getByRole("button", { name: "Try stop again" })).toBeVisible();
    await expect(notice.getByRole("button", { name: "Copy support details" })).toBeVisible();

    await context.setOffline(false);
    await notice.getByRole("button", { name: "Try stop again" }).click();
    await expect(page.getByRole("heading", { name: "BearVision is stopping safely" })).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
  } finally {
    await context.setOffline(false);
    await fixture.close();
  }
});

test("terminal recovery replaces a stale unconfirmed stop after reconnect", async ({ page, context }) => {
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    const runtime = fixture.runtimes[0];

    await context.setOffline(true);
    await page.getByRole("button", { name: "Stop runtime" }).click();
    await expect(page.getByRole("button", { name: "Try stop again" })).toBeVisible();
    runtime.emit("exit", 1, null);

    await context.setOffline(false);
    await expect(page.getByRole("button", { name: "Try stop again" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "End failed run" })).toBeVisible();
  } finally {
    await context.setOffline(false);
    await fixture.close();
  }
});

test("320 px view shows camera, queue, readiness and navigation without horizontal scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  const fixture = await startFixture();
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Run scenario" }).click();
    const runtime = fixture.runtimes[0];
    runtime.send("capture_activity_changed", {
      activity: "capturing", request_id: "capture-mobile", pending_captures: 1,
    }, 1);
    runtime.send("clip_job_updated", {
      job_id: "clip-mobile",
      status: "processing",
      state_changed_at_utc: "2026-09-09T08:00:00Z",
      counts: { queued: 0, processing: 1, failed: 0, completed: 0 },
    }, 2);

    const mobile = page.getByRole("region", { name: "Mobile operation status" });
    await expect(mobile).toBeVisible();
    await expect(mobile).toContainText("Camera");
    await expect(mobile).toContainText("Capturing");
    await expect(mobile).toContainText("Background clips");
    await expect(mobile).toContainText("1 active");
    await expect(mobile).toContainText("Simulation does not check physical equipment.");
    await expect(page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Logs" })).toBeVisible();
    await expect.poll(() => page.evaluate(
      () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
    )).toBe(true);
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

test("hardware start shows readiness checking while preflight runs", async ({ page }) => {
  let calls = 0;
  let finishStartPreflight;
  const fixture = await startFixture({
    runReadiness: async () => {
      calls += 1;
      if (calls === 1) return readinessReport({ checks: [] });
      return new Promise((resolve) => {
        finishStartPreflight = () => resolve(readinessReport({ checks: [] }));
      });
    },
  });
  try {
    await page.goto(fixture.url);
    await page.getByRole("button", { name: "Hardware" }).click();
    const summary = page.getByRole("region", { name: "Readiness summary" });
    await expect(summary).toContainText("Ready");
    await page.getByRole("button", { name: "Start hardware" }).click();
    await expect(summary).toContainText("Checking");
    await expect(page.getByText("Hardware ready", { exact: true })).toHaveCount(0);
    await expect.poll(() => Boolean(finishStartPreflight)).toBe(true);
    finishStartPreflight();
    await expect(page.getByRole("region", { name: "Pipeline" })).toContainText("Monitoring");
  } finally {
    await fixture.close();
  }
});
