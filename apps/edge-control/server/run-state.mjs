import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

const TRANSIENT_EVENT_KINDS = new Set(["preview_frame", "tracking_observation"]);
const ACTIVE_STAGES = new Set([
  "initializing", "monitoring", "recording", "post_processing", "packaging",
  "uploading", "stopping", "failed",
]);

function copy(value) {
  return value == null ? value : structuredClone(value);
}

function defaultData() {
  return {
    control_api_version: "2.0",
    mode: "simulation",
    active_run: null,
    recent_runs: [],
    sequence: 0,
  };
}

export class RunState {
  constructor({ stateFile = null, now = () => new Date().toISOString(), createId = null } = {}) {
    this.stateFile = stateFile;
    this.now = now;
    this.createId = createId ?? ((prefix) => `${prefix}-${randomUUID()}`);
    this.data = this.#load();
    this.#recoverInterruptedRun();
  }

  #load() {
    if (!this.stateFile || !existsSync(this.stateFile)) return defaultData();
    try {
      const parsed = JSON.parse(readFileSync(this.stateFile, "utf8"));
      if (parsed?.control_api_version !== "2.0") return defaultData();
      return {
        ...defaultData(),
        ...parsed,
        recent_runs: Array.isArray(parsed.recent_runs) ? parsed.recent_runs.slice(0, 10) : [],
      };
    } catch {
      return defaultData();
    }
  }

  #persist() {
    if (!this.stateFile) return;
    mkdirSync(dirname(this.stateFile), { recursive: true });
    const temporary = `${this.stateFile}.tmp`;
    writeFileSync(temporary, `${JSON.stringify(this.data, null, 2)}\n`, "utf8");
    renameSync(temporary, this.stateFile);
  }

  #recoverInterruptedRun() {
    const run = this.data.active_run;
    if (!run || run.process_state === "exited") return;
    const occurredAt = this.now();
    const failureId = `failure-${run.run_id}-control-restart`;
    run.process_state = "exited";
    run.stage = "failed";
    run.stage_started_at = occurredAt;
    run.stop_state = "none";
    run.failures = [{
      failure_id: failureId,
      operation_id: run.current_operation?.operation_id ?? null,
      stage: run.current_operation?.stage ?? "failed",
      component: "control_server",
      error: "The control server restarted while the runtime was active.",
      operator_message: "The runtime connection was lost when Edge Control restarted.",
      corrective_action: "Review the retained evidence, then Restart the runtime.",
      severity: "terminal",
      retryable: false,
      occurred_at: occurredAt,
      resolved_at: null,
      attempts: 1,
    }, ...run.failures.filter((failure) => failure.failure_id !== failureId)];
    this.#persist();
  }

  selectMode(mode) {
    if (!new Set(["simulation", "hardware"]).has(mode)) {
      throw new Error("mode must be 'simulation' or 'hardware'");
    }
    if (this.data.active_run && ACTIVE_STAGES.has(this.data.active_run.stage)) {
      throw new Error("mode cannot change while a runtime is active");
    }
    this.data.mode = mode;
    this.#persist();
    return this.snapshot();
  }

  start({ mode = this.data.mode, scenario = null, runId = null, restartOfRunId = null } = {}) {
    if (this.data.active_run && ACTIVE_STAGES.has(this.data.active_run.stage)) {
      throw new Error("a runtime is already active");
    }
    if (!new Set(["simulation", "hardware"]).has(mode)) {
      throw new Error("mode must be 'simulation' or 'hardware'");
    }
    const startedAt = this.now();
    this.data.mode = mode;
    this.data.active_run = {
      run_id: runId ?? this.createId("run"),
      restart_of_run_id: restartOfRunId,
      mode,
      scenario,
      stage: "initializing",
      stage_started_at: startedAt,
      started_at: startedAt,
      ended_at: null,
      current_operation: null,
      failures: [],
      artefacts: [],
      stop_state: "none",
      cleanup_status: "not_required",
      process_state: "starting",
      events: [],
      completion_reason: null,
    };
    this.#persist();
    return copy(this.data.active_run);
  }

  record(event) {
    const run = this.data.active_run;
    if (!run) return this.snapshot();
    const recorded = {
      ...copy(event),
      sequence: ++this.data.sequence,
      emitted_at: event.emitted_at ?? this.now(),
      run_id: run.run_id,
    };
    const payload = recorded.payload ?? {};

    if (recorded.kind === "lifecycle_changed") {
      this.#setStage(payload.stage, payload.operation_id ?? null);
    } else if (recorded.kind === "runtime_started") {
      run.process_state = "running";
      this.#setStage("monitoring", null);
    } else if (recorded.kind === "capture_started") {
      this.#setStage("recording", payload.operation_id ?? payload.request_id ?? payload.asset_id ?? null);
    } else if (recorded.kind === "finalize_clip") {
      this.#setStage("post_processing", payload.operation_id ?? payload.request_id ?? null);
    } else if (recorded.kind === "capture_completed") {
      this.#addArtefact("capture", payload);
    } else if (recorded.kind === "virtual_cameraman_completed") {
      this.#addArtefact("processed", {
        filename: payload.processed_filename,
        size_bytes: payload.processed_size_bytes,
      });
      this.#addArtefact("tracking", { filename: payload.tracking_filename });
      this.#addArtefact("debug", { filename: payload.debug_video_filename });
    } else if (recorded.kind === "clip_uploaded") {
      this.#setStage("monitoring", null);
    } else if (recorded.kind === "component_failed" || recorded.kind === "runtime_failed") {
      this.#addFailure(payload);
    } else if (recorded.kind === "failure_resolved") {
      const failure = run.failures.find((item) => item.failure_id === payload.failure_id);
      if (failure && failure.resolved_at == null) failure.resolved_at = this.now();
    }

    if (!TRANSIENT_EVENT_KINDS.has(recorded.kind)) {
      run.events = [recorded, ...run.events].slice(0, 500);
    }
    this.#persist();
    return recorded;
  }

  #setStage(stage, operationId) {
    if (!stage) return;
    const run = this.data.active_run;
    run.stage = stage;
    run.stage_started_at = this.now();
    run.current_operation = operationId ? { operation_id: operationId, stage } : null;
  }

  #addArtefact(kind, payload) {
    if (!payload?.filename) return;
    const run = this.data.active_run;
    const artefact = {
      kind,
      filename: payload.filename,
      size_bytes: payload.size_bytes ?? null,
      created_at: this.now(),
    };
    run.artefacts = [
      ...run.artefacts.filter((item) => !(item.kind === kind && item.filename === artefact.filename)),
      artefact,
    ];
  }

  #addFailure(payload) {
    const run = this.data.active_run;
    const failureId = payload.failure_id ?? this.createId("failure");
    const existing = run.failures.find((item) => item.failure_id === failureId);
    const failure = {
      failure_id: failureId,
      operation_id: payload.operation_id ?? run.current_operation?.operation_id ?? null,
      stage: payload.stage ?? run.stage,
      component: payload.component ?? "runtime",
      error: payload.error ?? payload.message ?? "Runtime failed",
      operator_message: payload.operator_message ?? payload.message ?? payload.error ?? "Runtime failed.",
      corrective_action: payload.corrective_action ?? "Review the technical details, then restart the runtime.",
      severity: payload.severity ?? "terminal",
      retryable: payload.retryable === true,
      occurred_at: existing?.occurred_at ?? this.now(),
      resolved_at: null,
      attempts: existing ? existing.attempts + 1 : 1,
    };
    run.failures = [failure, ...run.failures.filter((item) => item.failure_id !== failureId)];
    if (failure.severity !== "warning") this.#setStage("failed", failure.operation_id);
  }

  resolveFailure(failureId) {
    const run = this.data.active_run;
    const failure = run?.failures.find((item) => item.failure_id === failureId);
    if (!failure) throw new Error("unknown failure");
    failure.resolved_at = this.now();
    if (!run.failures.some((item) => item.resolved_at == null && item.severity !== "warning")) {
      this.#setStage("monitoring", null);
    }
    this.#persist();
    return copy(failure);
  }

  retryFailure(failureId) {
    const run = this.data.active_run;
    const failure = run?.failures.find((item) => item.failure_id === failureId);
    if (!failure) throw new Error("unknown failure");
    if (failure.resolved_at) throw new Error("failure is already resolved");
    if (!failure.retryable) throw new Error("failure is not retryable");
    this.#setStage(failure.stage, failure.operation_id);
    this.#persist();
    return copy(failure);
  }

  requestStop() {
    if (!this.data.active_run) throw new Error("no runtime is active");
    this.#setStage("stopping", this.data.active_run.current_operation?.operation_id ?? null);
    this.data.active_run.stop_state = "graceful_requested";
    this.data.active_run.process_state = "stopping";
    this.#persist();
  }

  allowForceStop() {
    if (!this.data.active_run || this.data.active_run.stop_state !== "graceful_requested") {
      throw new Error("graceful stop has not been requested");
    }
    this.data.active_run.stop_state = "force_available";
    this.#persist();
  }

  forceStop() {
    if (!this.data.active_run || this.data.active_run.stop_state !== "force_available") {
      throw new Error("force stop is not available");
    }
    this.data.active_run.stop_state = "forced";
    this.data.active_run.process_state = "stopping";
    this.data.active_run.cleanup_status = this.data.active_run.artefacts.length > 0
      ? "partial_artefacts_retained"
      : "no_artefacts";
    this.#persist();
  }

  complete(reason = "completed") {
    const run = this.data.active_run;
    if (!run) return null;
    run.stage = reason === "completed" ? "completed" : reason === "stopped" ? "stopped" : "failed";
    run.ended_at = this.now();
    run.stage_started_at = run.ended_at;
    run.current_operation = null;
    run.completion_reason = reason;
    run.process_state = "exited";
    this.data.recent_runs = [copy(run), ...this.data.recent_runs].slice(0, 10);
    this.data.active_run = null;
    this.#persist();
    return copy(run);
  }

  snapshot() {
    const active = this.data.active_run;
    return copy({
      control_api_version: "2.0",
      mode: this.data.mode,
      phase: active?.stage ?? "idle",
      scenario: active?.scenario ?? null,
      started_at: active?.started_at ?? null,
      last_event: active?.events?.[0] ?? null,
      active_run: active,
      recent_runs: this.data.recent_runs,
      sequence: this.data.sequence,
    });
  }

  setProcessState(processState) {
    if (!this.data.active_run) throw new Error("no runtime is active");
    if (!new Set(["starting", "running", "stopping", "exited"]).has(processState)) {
      throw new Error("invalid runtime process state");
    }
    this.data.active_run.process_state = processState;
    this.#persist();
    return this.snapshot();
  }
}
