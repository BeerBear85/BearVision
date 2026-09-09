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
  const summary = deriveOperatorSummary(state, { hardwareReady, missingWarnings });
  return {
    run,
    summary,
    unresolvedFailures,
    resolvedFailures: (run?.failures ?? []).filter((failure) => failure.resolved_at),
    missingWarnings,
    canStart: !run && (state.mode === "simulation" || hardwareReady),
    canStop: Boolean(run && ["starting", "running"].includes(run.process_state)),
    canForceStop: run?.stop_state === "force_available",
    canRestart: Boolean(run && run.stage === "failed" && run.process_state === "exited"),
    canEndFailedRun: Boolean(run && run.stage === "failed" && run.process_state === "exited"),
  };
}

export function deriveOperatorSummary(
  state,
  { hardwareReady = false, missingWarnings = [] } = {},
) {
  const run = state.active_run ?? null;
  if (run) {
    if (run.stage === "failed") {
      return {
        code: "needs_attention",
        label: "Action required",
        headline: "BearVision needs attention",
        explanation: "Review the problem below before continuing.",
        tone: "attention",
        requiresAction: true,
      };
    }
    if (run.stage === "stopping" || run.process_state === "stopping") {
      return {
        code: "stopping",
        label: "Stopping",
        headline: "BearVision is stopping safely",
        explanation: "Wait until the status says Stopped before disconnecting equipment.",
        tone: "working",
        requiresAction: false,
      };
    }
    if (["stopped", "completed"].includes(run.stage) && run.process_state !== "exited") {
      return {
        code: "finishing",
        label: "Finishing",
        headline: "Recording stopped — finishing the run",
        explanation: "BearVision is completing the remaining run cleanup.",
        tone: "working",
        requiresAction: false,
      };
    }
    if (run.stage === "initializing" || run.process_state === "starting") {
      return {
        code: "starting",
        label: "Starting",
        headline: "BearVision is starting",
        explanation: "Preparing the selected runtime. No action is needed yet.",
        tone: "working",
        requiresAction: false,
      };
    }
    if (run.capture_activity?.activity === "capturing") {
      return {
        code: "recording",
        label: "Recording",
        headline: "BearVision is recording riders",
        explanation: "New clips are being captured while earlier clips can finish in the background.",
        tone: "working",
        requiresAction: false,
      };
    }
    return {
      code: "running",
      label: "Running",
      headline: "BearVision is running normally",
      explanation: "Monitoring is active. No operator action is needed.",
      tone: "ok",
      requiresAction: false,
    };
  }

  if (state.mode === "simulation") {
    return {
      code: "ready",
      label: "Simulation ready",
      headline: "Simulation is ready to run",
      explanation: "Physical equipment is not checked in simulation mode.",
      tone: "ok",
      requiresAction: false,
    };
  }
  if (!state.readiness || state.readiness.status === "not_checked") {
    return {
      code: "needs_attention",
      label: "Check required",
      headline: "Check the hardware before starting",
      explanation: "Run readiness to verify the camera, scanner and required services.",
      tone: "attention",
      requiresAction: true,
    };
  }
  if (state.readiness.blocking) {
    return {
      code: "needs_attention",
      label: "Start blocked",
      headline: "Hardware is not ready",
      explanation: "Resolve the blocking readiness issue shown below, then check again.",
      tone: "attention",
      requiresAction: true,
    };
  }
  if (missingWarnings.length > 0) {
    return {
      code: "needs_attention",
      label: "Review warnings",
      headline: "Review the hardware warnings",
      explanation: "Acknowledge each warning before starting BearVision.",
      tone: "attention",
      requiresAction: true,
    };
  }
  return {
    code: "ready",
    label: "Hardware ready",
    headline: "BearVision is ready to start",
    explanation: hardwareReady
      ? "Hardware readiness passed."
      : "Complete readiness before starting.",
    tone: hardwareReady ? "ok" : "attention",
    requiresAction: !hardwareReady,
  };
}

export function failurePresentation(failure, mode) {
  if (failure?.component === "camera") {
    return {
      headline: mode === "simulation"
        ? "The simulated camera could not record."
        : "The camera could not record.",
      impact: mode === "simulation"
        ? "No new test clips are being captured."
        : "No new clips are being captured while the camera is unavailable.",
      correctiveAction: mode === "simulation"
        ? "Restart the runtime to continue the test."
        : "Check the camera power and connection, then restart BearVision. Contact support if the failure returns.",
    };
  }
  if (failure?.component === "control_server") {
    return {
      headline: "Edge Control lost the previous runtime.",
      impact: mode === "simulation"
        ? "The previous test can no longer be controlled. Its evidence is retained."
        : "Edge Control cannot confirm whether the previous runtime is still recording.",
      correctiveAction: mode === "simulation"
        ? "End the failed run, then start another test when ready."
        : "Check the camera and runtime first. When they are stopped, end the failed run or contact support.",
    };
  }
  return {
    headline: failure?.operator_message ?? "The runtime operation failed.",
    impact: failure?.operator_impact ?? null,
    correctiveAction: failure?.corrective_action ?? "Review the technical details and contact support.",
  };
}

function stateLabel(run) {
  if (!run) return "Idle";
  return `${run.stage ?? "unknown"} / ${run.process_state ?? "unknown"}`;
}

export function actionFailureNotice(action, error, context = {}) {
  const occurredAt = context.occurredAt ?? new Date().toISOString();
  const run = context.run ?? null;
  const lastKnownAt = context.lastKnownAt ?? null;
  const isUnconfirmedStop = action === "stop" && (
    error?.name === "TypeError"
    || error?.code === "STREAM_DISCONNECTED"
    || /fetch|network/i.test(error?.message ?? "")
  );
  const supportDetails = [
    `Action: ${action}`,
    `Result: ${isUnconfirmedStop ? "unconfirmed" : "failed"}`,
    `Occurred: ${occurredAt}`,
    `Run: ${run?.run_id ?? "none"}`,
    `Mode: ${context.mode ?? run?.mode ?? "unknown"}`,
    `Scenario: ${run?.scenario ?? context.scenario ?? "none"}`,
    `Last known runtime state: ${stateLabel(run)}`,
    `Last snapshot: ${lastKnownAt ?? "unknown"}`,
    `Control connection: ${context.streamConnected ? "live" : "disconnected"}`,
    `Error code: ${error?.code ?? "none"}`,
    `Technical error: ${error?.message ?? "Unknown error"}`,
  ].join("\n");

  if (isUnconfirmedStop) {
    return {
      action,
      runId: run?.run_id ?? null,
      status: "unconfirmed",
      eyebrow: "Action not confirmed",
      headline: "We could not confirm that BearVision stopped",
      message: "Check the control connection, then try again. BearVision may still be recording.",
      lastKnown: `Last known state: ${stateLabel(run)}`,
      occurredAt,
      retryable: true,
      supportDetails,
    };
  }
  return {
    action,
    runId: run?.run_id ?? null,
    status: "failed",
    eyebrow: "Action failed",
    headline: error?.message ?? "The action failed.",
    message: error?.correctiveAction ?? "Try again. Contact support if the problem continues.",
    lastKnown: `Last known state: ${stateLabel(run)}`,
    occurredAt,
    retryable: false,
    supportDetails,
  };
}

export function reconcileActionNotice(notice, state, streamConnected) {
  if (!notice || notice.status !== "unconfirmed" || notice.action !== "stop") return notice;
  if (!streamConnected || state?.phase === "loading") return notice;
  const run = state?.active_run ?? null;
  if (!run || run.run_id !== notice.runId) return null;
  if (run.stage === "failed" || run.process_state === "exited") return null;
  if (run.stage === "stopping" || run.process_state === "stopping") return null;
  return notice;
}

export function mediaFailureNotice(kind, error, context = {}) {
  const run = context.run ?? null;
  const completed = run?.stage === "completed";
  const label = kind === "tracking" ? "tracking view" : "media view";
  return {
    kind,
    headline: completed
      ? `Run completed — ${label} unavailable`
      : `${label.charAt(0).toUpperCase()}${label.slice(1)} unavailable`,
    message: completed
      ? "The run result remains Completed. Other media views may still be available."
      : "Other media views may still be available.",
    correctiveAction: `Try loading the ${label} again. Contact support if it remains unavailable.`,
    supportDetails: [
      `Media: ${kind}`,
      `Run: ${run?.run_id ?? "none"}`,
      `Run result: ${run?.stage ?? "unknown"}`,
      `Filename: ${context.filename ?? "unknown"}`,
      `Error code: ${error?.code ?? "none"}`,
      `Technical error: ${error?.message ?? "Unknown error"}`,
    ].join("\n"),
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
    run_id: run?.run_id ?? null,
    scenario: run?.scenario ?? null,
    captured_at: capture?.created_at ?? processed?.created_at ?? null,
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
