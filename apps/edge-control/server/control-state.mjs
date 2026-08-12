export class ControlState {
  constructor() {
    this.mode = "simulation";
    this.phase = "idle";
    this.scenario = null;
    this.startedAt = null;
    this.lastEvent = null;
    this.sequence = 0;
  }

  selectMode(mode) {
    if (!new Set(["simulation", "hardware"]).has(mode)) {
      throw new Error("mode must be 'simulation' or 'hardware'");
    }
    if (this.phase !== "idle") {
      throw new Error("mode cannot change while a runtime is active");
    }
    this.mode = mode;
  }

  start({ mode, scenario = null }) {
    if (this.phase !== "idle") throw new Error("a runtime is already active");
    this.mode = mode;
    this.phase = "starting";
    this.scenario = scenario;
    this.startedAt = new Date().toISOString();
    this.lastEvent = null;
  }

  record(event) {
    this.sequence += 1;
    this.lastEvent = { sequence: this.sequence, ...event };
    if (event.kind === "runtime_started") this.phase = "running";
  }

  stop(reason = "completed") {
    this.phase = "idle";
    this.lastEvent = { sequence: ++this.sequence, kind: "runtime_stopped", payload: { reason } };
    this.scenario = null;
    this.startedAt = null;
  }

  snapshot() {
    return {
      control_api_version: "1.0",
      mode: this.mode,
      phase: this.phase,
      scenario: this.scenario,
      started_at: this.startedAt,
      last_event: this.lastEvent,
    };
  }
}
