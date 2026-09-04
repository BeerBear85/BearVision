import { createInterface } from "node:readline";

import { createRuntimeLogLevelClassifier } from "../src/log-level.js";
import { parseRuntimeEventLine } from "./runtime-events.mjs";

const SNAPSHOTLESS_EVENT_KINDS = new Set([
  "preview_frame",
  "runtime_log",
  "tracking_observation",
]);

export class RuntimeSupervisor {
  constructor({
    state,
    spawnRuntime,
    validateStart = async () => {},
    publish = () => {},
    stopTimeoutMs = 10_000,
    setTimer = setTimeout,
    clearTimer = clearTimeout,
  }) {
    this.state = state;
    this.spawnRuntime = spawnRuntime;
    this.validateStart = validateStart;
    this.publish = publish;
    this.stopTimeoutMs = stopTimeoutMs;
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.child = null;
    this.stopTimer = null;
    this.lastStart = null;
  }

  async start({ mode, scenario = null, acknowledgedWarnings = [], restartOfRunId = null }) {
    await this.validateStart({ mode, scenario, acknowledgedWarnings });
    const run = this.state.start({ mode, scenario, restartOfRunId });
    this.lastStart = { mode, scenario };
    try {
      this.child = this.spawnRuntime({ mode, scenario, runId: run.run_id });
      this.#attachOutput(this.child.stdout, "stdout");
      this.#attachOutput(this.child.stderr, "stderr");
      this.child.once("error", (error) => this.#handleProcessError(error));
      this.child.once("exit", (code, signal) => this.#handleExit(code, signal));
      this.#record({
        kind: "runtime_started",
        payload: { mode, scenario, pid: this.child.pid },
      });
      return run;
    } catch (error) {
      this.child = null;
      this.state.setProcessState("exited");
      this.#record({
        kind: "runtime_failed",
        payload: {
          component: "runtime_process",
          error: error.message,
          operator_message: "The Edge runtime could not be started.",
          corrective_action: "Review readiness and technical details, then restart the runtime.",
          severity: "terminal",
          retryable: false,
        },
      });
      throw error;
    }
  }

  #attachOutput(stream, source) {
    if (!stream) return;
    const lines = createInterface({ input: stream });
    const classifyLogLevel = createRuntimeLogLevelClassifier();
    lines.on("line", (line) => {
      if (!line.trim()) return;
      try {
        this.#record(parseRuntimeEventLine(line));
      } catch {
        this.#record({
          kind: "runtime_log",
          payload: { source, level: classifyLogLevel(line), message: line },
        });
      }
    });
  }

  #record(event) {
    const recorded = this.state.record(event);
    const snapshot = SNAPSHOTLESS_EVENT_KINDS.has(recorded.kind)
      ? null
      : this.state.snapshot();
    this.publish(recorded, snapshot);
    return recorded;
  }

  #handleProcessError(error) {
    if (!this.state.snapshot().active_run) return;
    this.child = null;
    this.state.setProcessState("exited");
    this.#record({
      kind: "runtime_failed",
      payload: {
        component: "runtime_process",
        error: error.message,
        operator_message: "The Edge runtime process failed.",
        corrective_action: "Review the technical details, then restart the runtime.",
        severity: "terminal",
        retryable: false,
      },
    });
  }

  #handleExit(code, signal) {
    if (this.stopTimer != null) {
      this.clearTimer(this.stopTimer);
      this.stopTimer = null;
    }
    const snapshot = this.state.snapshot();
    const run = snapshot.active_run;
    this.child = null;
    if (!run) return;
    this.state.setProcessState("exited");
    const stoppedByOperator = run.stop_state !== "none";
    if (code === 0 || stoppedByOperator) {
      this.#record({
        kind: "runtime_completed",
        payload: { code, signal, reason: stoppedByOperator ? "stopped" : "completed" },
      });
      const completed = this.state.complete(stoppedByOperator ? "stopped" : "completed");
      this.publish({
        kind: "run_archived",
        payload: { run_id: completed.run_id, reason: completed.completion_reason },
      }, this.state.snapshot());
      return;
    }
    const terminalFailureReported = this.state.snapshot().active_run.failures.some(
      (failure) => failure.resolved_at == null && failure.severity === "terminal",
    );
    if (terminalFailureReported) {
      this.#record({
        kind: "runtime_exit_observed",
        payload: { code, signal },
      });
      return;
    }
    this.#record({
      kind: "runtime_failed",
      payload: {
        component: "runtime_process",
        error: `Runtime exited with code ${code ?? "unknown"}${signal ? ` (${signal})` : ""}`,
        operator_message: "The Edge runtime stopped unexpectedly.",
        corrective_action: "Review the failed stage and technical details, then restart the runtime.",
        severity: "terminal",
        retryable: false,
      },
    });
  }

  #assertRun(runId) {
    const run = this.state.snapshot().active_run;
    if (!run || run.run_id !== runId) throw new Error("run is not active");
    return run;
  }

  #writeCommand(command) {
    if (!this.child?.stdin?.writable) {
      throw new Error("runtime process command channel is not available");
    }
    this.child.stdin.write(`${JSON.stringify(command)}\n`);
  }

  stop(runId) {
    const run = this.#assertRun(runId);
    if (!this.child) throw new Error("runtime process is not active");
    if (run.mode === "hardware" && !this.child.stdin?.writable) {
      throw new Error("runtime process command channel is not available");
    }
    this.state.requestStop();
    this.#record({ kind: "stop_requested", payload: { pid: this.child.pid } });
    if (run.mode === "hardware") {
      this.#writeCommand({ command_version: "1.0", kind: "stop_runtime" });
    } else {
      this.child.kill("SIGTERM");
    }
    this.stopTimer = this.setTimer(() => {
      const run = this.state.snapshot().active_run;
      if (!run || run.run_id !== runId || run.stop_state !== "graceful_requested") return;
      this.state.allowForceStop();
      this.#record({
        kind: "force_stop_available",
        payload: { timeout_ms: this.stopTimeoutMs },
      });
    }, this.stopTimeoutMs);
    return this.state.snapshot();
  }

  forceStop(runId) {
    this.#assertRun(runId);
    if (!this.child) throw new Error("runtime process is not active");
    this.state.forceStop();
    this.#record({ kind: "force_stop_requested", payload: { pid: this.child.pid } });
    this.child.kill("SIGKILL");
    return this.state.snapshot();
  }

  retry(runId, failureId) {
    this.#assertRun(runId);
    if (!this.child?.stdin?.writable) throw new Error("runtime process is not available for retry");
    const failure = this.state.retryFailure(failureId);
    const command = {
      command_version: "1.0",
      kind: "retry_failure",
      failure_id: failure.failure_id,
    };
    this.#writeCommand(command);
    this.#record({ kind: "retry_requested", payload: {
      failure_id: failure.failure_id,
      operation_id: failure.operation_id,
    } });
    return this.state.snapshot();
  }

  async restart(runId) {
    const run = this.#assertRun(runId);
    if (this.child) throw new Error("stop the active runtime before restarting");
    const start = { mode: run.mode, scenario: run.scenario };
    this.state.complete("failed");
    return this.start({ ...start, restartOfRunId: run.run_id });
  }

  shutdown() {
    if (this.stopTimer != null) this.clearTimer(this.stopTimer);
    if (!this.child) return;
    const run = this.state.snapshot().active_run;
    if (run?.mode === "hardware" && this.child.stdin?.writable) {
      this.#writeCommand({ command_version: "1.0", kind: "stop_runtime" });
    } else {
      this.child.kill("SIGTERM");
    }
  }
}
