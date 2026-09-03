export function runtimeArguments({
  mode,
  runId,
  scenarioPath = null,
  configPath,
  captureRoot,
  scratchRoot,
  localQueueRoot,
}) {
  if (typeof runId !== "string" || !runId) throw new Error("run id is required");
  if (mode === "simulation") {
    if (!scenarioPath) throw new Error("simulation scenario path is required");
    return [
      "-m", "bearvision.control", "simulate", scenarioPath,
      "--run-id", runId, "--realtime",
      "--local-queue-root", localQueueRoot, "--config", configPath,
    ];
  }
  if (mode === "hardware") {
    return [
      "-m", "bearvision.control", "hardware", "--config", configPath,
      "--run-id", runId, "--capture-dir", captureRoot, "--scratch-dir", scratchRoot,
    ];
  }
  throw new Error(`unsupported runtime mode: ${mode}`);
}
