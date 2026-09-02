export class ControlError extends Error {
  constructor(code, message, { status = 400, correctiveAction = null, details = null } = {}) {
    super(message);
    this.name = "ControlError";
    this.code = code;
    this.status = status;
    this.correctiveAction = correctiveAction;
    this.details = details;
  }
}

export class ReadinessService {
  constructor({ runCommand }) {
    this.runCommand = runCommand;
    this.report = null;
  }

  async run() {
    const report = await this.runCommand();
    if (
      report?.readiness_schema_version !== "1.0"
      || typeof report.blocking !== "boolean"
      || !Array.isArray(report.warning_ids)
      || !Array.isArray(report.checks)
    ) {
      throw new ControlError(
        "READINESS_INVALID",
        "The runtime returned an invalid readiness report.",
        {
          status: 502,
          correctiveAction: "Review the Python runtime logs and readiness contract.",
        },
      );
    }
    this.report = structuredClone(report);
    return this.current();
  }

  current() {
    return this.report ? structuredClone(this.report) : null;
  }

  async assertReady({ acknowledgedWarnings = [] } = {}) {
    await this.run();
    if (this.report.blocking) {
      throw new ControlError(
        "READINESS_BLOCKED",
        "Hardware cannot start while critical readiness checks are failing.",
        {
          status: 409,
          correctiveAction: "Resolve every critical failure and run readiness again.",
          details: this.current(),
        },
      );
    }
    const acknowledged = new Set(acknowledgedWarnings);
    const missing = this.report.warning_ids.filter((warningId) => !acknowledged.has(warningId));
    if (missing.length > 0) {
      throw new ControlError(
        "READINESS_WARNING_ACKNOWLEDGEMENT_REQUIRED",
        "Acknowledge every readiness warning before starting hardware.",
        {
          status: 409,
          correctiveAction: "Review the warnings and explicitly acknowledge them.",
          details: { missing_warning_ids: missing, report: this.current() },
        },
      );
    }
    return this.current();
  }
}
