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
  constructor({ runCommand, onChange = () => {} }) {
    this.runCommand = runCommand;
    this.onChange = onChange;
    this.report = null;
  }

  setCurrent(report) {
    this.report = structuredClone(report);
    this.onChange(this.current());
  }

  setFailure(error) {
    this.setCurrent({
      readiness_schema_version: "1.0",
      status: "failed",
      blocking: true,
      warning_ids: [],
      checks: [],
      failure: {
        code: error.code ?? "READINESS_FAILED",
        message: error.message ?? "Hardware readiness failed.",
        corrective_action: error.correctiveAction ?? null,
        details: error.details ?? null,
      },
    });
  }

  async run() {
    this.setCurrent({
      readiness_schema_version: "1.0",
      status: "checking",
      blocking: true,
      warning_ids: [],
      checks: [],
      failure: null,
    });
    let report;
    try {
      report = await this.runCommand();
    } catch (error) {
      this.setFailure(error);
      throw error;
    }
    if (
      report?.readiness_schema_version !== "1.0"
      || typeof report.blocking !== "boolean"
      || !Array.isArray(report.warning_ids)
      || !Array.isArray(report.checks)
    ) {
      const error = new ControlError(
        "READINESS_INVALID",
        "The runtime returned an invalid readiness report.",
        {
          status: 502,
          correctiveAction: "Review the Python runtime logs and readiness contract.",
        },
      );
      this.setFailure(error);
      throw error;
    }
    this.setCurrent({
      ...report,
      status: report.blocking ? "blocked" : "ready",
      failure: null,
    });
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
