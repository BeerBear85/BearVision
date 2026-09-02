export const PIPELINE_STAGES = [
  ["readiness", "Ready"],
  ["monitoring", "Monitoring"],
  ["recording", "Recording"],
  ["post_processing", "Processing"],
  ["packaging", "Packaging"],
  ["uploading", "Uploading"],
  ["complete", "Complete"],
];

export function pipelineForStage(stage, failedStage = null) {
  const normalized = stage === "completed" ? "complete" : stage;
  const effective = normalized === "failed" ? failedStage : normalized;
  const currentIndex = PIPELINE_STAGES.findIndex(([key]) => key === effective);
  return PIPELINE_STAGES.map(([key, label], index) => ({
    key,
    label,
    status: currentIndex < 0
      ? "upcoming"
      : index < currentIndex
        ? "complete"
        : index === currentIndex
          ? normalized === "failed" ? "failed" : "current"
          : "upcoming",
  }));
}

export function deriveOperatorView(state, acknowledgedWarnings = new Set()) {
  const run = state.active_run ?? null;
  const readiness = state.readiness ?? null;
  const unresolvedFailures = (run?.failures ?? []).filter((failure) => !failure.resolved_at);
  const missingWarnings = (readiness?.warning_ids ?? []).filter(
    (warningId) => !acknowledgedWarnings.has(warningId),
  );
  const hardwareReady = Boolean(readiness && !readiness.blocking && missingWarnings.length === 0);
  return {
    run,
    unresolvedFailures,
    resolvedFailures: (run?.failures ?? []).filter((failure) => failure.resolved_at),
    missingWarnings,
    canStart: !run && (state.mode === "simulation" || hardwareReady),
    canStop: Boolean(run && ["starting", "running"].includes(run.process_state)),
    canForceStop: run?.stop_state === "force_available",
    canRestart: Boolean(run && run.stage === "failed" && run.process_state === "exited"),
  };
}

function artefact(run, kind) {
  return run?.artefacts?.find((item) => item.kind === kind) ?? null;
}

function mediaUrl(item) {
  return item?.filename ? `/api/captures/${encodeURIComponent(item.filename)}` : null;
}

export function restoreCapturedClip(run) {
  const capture = artefact(run, "capture");
  const processed = artefact(run, "processed");
  const debug = artefact(run, "debug");
  const tracking = artefact(run, "tracking");
  if (!capture && !processed) return null;
  return {
    filename: capture?.filename ?? processed?.filename,
    size_bytes: capture?.size_bytes ?? null,
    url: mediaUrl(capture),
    processed_filename: processed?.filename ?? null,
    processed_size_bytes: processed?.size_bytes ?? null,
    processed_url: mediaUrl(processed),
    debug_filename: debug?.filename ?? null,
    debug_url: mediaUrl(debug),
    tracking_filename: tracking?.filename ?? null,
    tracking_url: mediaUrl(tracking),
  };
}
