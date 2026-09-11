import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import bearVisionLogo from "../../../logo/Logo.svg";
import {
  appendRetainedTraceEvent,
  runtimeLogLevel,
  showsAtMinimumLogLevel,
} from "./log-level.js";
import {
  actionFailureNotice,
  deriveOperatorView,
  failurePresentation,
  mediaFailureNotice,
  reconcileActionNotice,
  restoreCapturedClip,
  stopOutcomePresentation,
} from "./operator-model.js";
import "./styles.css";

const initialState = {
  control_api_version: "2.0",
  mode: "hardware",
  phase: "loading",
  active_run: null,
  recent_runs: [],
  readiness: null,
};

const DEFAULT_SIMULATION_SCENARIO = "wakeboard-testmovie1-yolo.yaml";

const PAGE_PATHS = {
  overview: "/",
  readiness: "/readiness",
  logs: "/logs",
};

function pageFromPath(pathname) {
  if (pathname === PAGE_PATHS.readiness) return "readiness";
  if (pathname === PAGE_PATHS.logs || pathname === "/diagnostics") return "logs";
  return "overview";
}

function preferredScenarioName(scenarios) {
  return scenarios.find((scenario) => scenario.name === DEFAULT_SIMULATION_SCENARIO)?.name
    ?? scenarios.find((scenario) => scenario.video_url)?.name
    ?? scenarios[0]?.name
    ?? "";
}

function startupGuidance(mode, scenario) {
  if (mode === "hardware") {
    return "Connecting to the GoPro and starting BearTag monitoring. Wait until the status says Running.";
  }
  if (!scenario?.video_url) return "Preparing the simulation. This usually takes a few seconds.";
  const duration = Number.isFinite(Number(scenario.duration_s))
    ? `${Math.ceil(Number(scenario.duration_s))}-second `
    : "";
  return `Loading and analysing the ${duration}input test video. The first run can take up to a minute before replay starts.`;
}

function formatLabel(value) {
  if (!value) return "Connecting";
  const label = value.replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function formatFileSize(bytes) {
  return Number.isFinite(Number(bytes))
    ? `${Math.round(Number(bytes) / 1024)} KiB`
    : "Size unavailable";
}

function formatDate(value) {
  if (!value) return "Time unavailable";
  return new Intl.DateTimeFormat("da-DK", {
    dateStyle: "medium",
    timeStyle: "short",
    hour12: false,
  }).format(new Date(value));
}

function elapsedSince(value, now) {
  if (!value) return "";
  const seconds = Math.max(0, Math.floor((now - new Date(value).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function eventMessage(event) {
  const logLevel = runtimeLogLevel(event);
  if (logLevel) return `${formatLabel(logLevel)} log`;
  const labels = {
    mode_selected: "Runtime mode changed",
    readiness_updated: "Readiness updated",
    runtime_started: "Runtime started",
    runtime_completed: "Runtime completed",
    runtime_failed: "Runtime failed",
    lifecycle_changed: "Pipeline stage changed",
    person_detected: "Vision detected a person",
    capture_started: "GoPro capture started",
    capture_completed: "Capture completed",
    clip_uploaded: "Clip uploaded",
    component_failed: "Component failure",
    failure_resolved: "Failure resolved",
    retry_requested: "Retry requested",
    stop_requested: "Stop requested",
    force_stop_available: "Force stop available",
    force_stop_requested: "Force stop requested",
    preview_frame: "Preview frame analysed",
    virtual_cameraman_completed: "Virtual cameraman completed",
    tracking_observation: "Rider position estimated",
    clip_queue_snapshot: "Clip queue restored",
    clip_job_updated: "Clip job updated",
    capture_activity_changed: "Camera activity changed",
  };
  return labels[event.kind] ?? event.kind?.replaceAll("_", " ") ?? "Event";
}

function Indicator({ label, status = "idle", detail }) {
  return (
    <div className="indicator">
      <span className={`dot ${status}`} aria-hidden="true" />
      <span><strong>{label}</strong><small>{detail}</small></span>
    </div>
  );
}

function connectionLabel(connectionState) {
  if (connectionState === "loading") return "Connecting";
  if (connectionState === "reconnecting") return "Reconnecting";
  return "Live";
}

function OperatorOverview({ connectionState, mode, run, summary }) {
  const queue = run?.clip_queue?.counts ?? {};
  const camera = run?.capture_activity ?? { activity: "idle", pending_captures: 0 };
  return (
    <section className={`operator-overview ${summary.tone}`} aria-labelledby="operator-status-heading" aria-live="polite">
      <div>
        <span className="eyebrow">{summary.requiresAction ? "Action required" : "Operator status"}</span>
        <h2 id="operator-status-heading">{summary.headline}</h2>
        <p>{summary.explanation}</p>
        <div className="operator-facts">
          <span><strong>Camera</strong> · {formatLabel(camera.activity)}</span>
          <span><strong>Background</strong> · {queue.processing ?? 0} active, {queue.queued ?? 0} queued</span>
          <span><strong>Connection</strong> · {connectionLabel(connectionState)}</span>
          <span><strong>Mode</strong> · {formatLabel(mode)}</span>
        </div>
      </div>
      <span className={`status-badge ${summary.tone}`}><span className="status-dot" />{summary.label}</span>
    </section>
  );
}

function StopOutcome({ outcome, onDismiss }) {
  return (
    <div className="stop-outcome" role="status">
      <div>
        <span className="eyebrow">Stop completed</span>
        <strong>{outcome.headline}</strong>
        <p>{outcome.message}</p>
      </div>
      <button className="dismiss-stop-outcome" type="button" aria-label="Dismiss stop result" onClick={onDismiss}>×</button>
    </div>
  );
}

function ActionNotice({ notice, copied, onCopy, onDismiss, onRetry }) {
  return (
    <div className="action-notice" role="alert">
      <div>
        <span className="eyebrow">{notice.eyebrow}</span>
        <strong>{notice.headline}</strong>
        <p>{notice.message}</p>
        <small>{notice.lastKnown} · {formatDate(notice.occurredAt)}</small>
        <details>
          <summary>Technical details</summary>
          <pre>{notice.supportDetails}</pre>
        </details>
      </div>
      <div className="action-notice-actions">
        <button className="secondary" type="button" onClick={onCopy}>{copied ? "Copied" : "Copy support details"}</button>
        {notice.retryable && <button className="danger" type="button" onClick={onRetry}>Try stop again</button>}
        <button className="dismiss-notice" type="button" aria-label="Dismiss message" onClick={onDismiss}>×</button>
      </div>
    </div>
  );
}

function MediaIssue({ issue, onRetry }) {
  return (
    <div className="media-issue" role="status">
      <div>
        <strong>{issue.headline}</strong>
        <p>{issue.message}</p>
        <p>{issue.correctiveAction}</p>
        <details>
          <summary>Technical details</summary>
          <pre>{issue.supportDetails}</pre>
        </details>
      </div>
      <button className="secondary" type="button" onClick={onRetry}>Try tracking again</button>
    </div>
  );
}

function Pipeline({ mode, run, readiness, readinessChecking, summary, now }) {
  const queue = run?.clip_queue ?? {
    counts: { queued: 0, processing: 0, failed: 0, completed: 0 },
    current_job: null,
    oldest_queued_at_utc: null,
    jobs: [],
  };
  const camera = run?.capture_activity ?? {
    activity: "idle", request_id: null, pending_captures: 0,
  };
  const activeJob = queue.jobs?.find((job) => job.job_id === queue.current_job) ?? null;
  const backgroundStages = [
    "queued", "processing", "packaging", "uploading", "failed", "completed",
  ];
  const monitoringStatus = summary.code === "starting"
    ? "upcoming"
    : run?.stage === "monitoring"
      ? "current"
      : run
        ? "complete"
        : "upcoming";
  return (
    <section className="pipeline panel" aria-labelledby="pipeline-heading" aria-live="polite">
      <div className="pipeline-heading">
        <div>
          <span className="eyebrow">Live operation</span>
          <h2 id="pipeline-heading">Concurrent pipeline</h2>
        </div>
        <span className={`status-badge ${summary.tone}`}>
          <span className="status-dot" />
          {summary.label}
          {run?.stage_started_at && <small>{elapsedSince(run.stage_started_at, now)}</small>}
        </span>
      </div>
      <div className="operation-tracks">
        <article className="operation-track" aria-label="Live track">
          <h3>Live</h3>
          <ol className="pipeline-steps">
            <li className={readinessChecking ? "current" : readiness?.blocking ? "failed" : "complete"} aria-current={readinessChecking ? "step" : undefined}><span>1</span><strong>{mode === "simulation" ? "Readiness: Not used" : readinessChecking ? "Readiness: Checking" : "Readiness"}</strong></li>
            <li className={monitoringStatus} aria-current={monitoringStatus === "current" ? "step" : undefined}><span>2</span><strong>Monitoring</strong></li>
            <li className={camera.activity === "capturing" ? "current" : "upcoming"}><span>3</span><strong>Camera: {formatLabel(camera.activity)}</strong></li>
          </ol>
          <p className="pipeline-detail">{camera.pending_captures} pending capture{camera.pending_captures === 1 ? "" : "s"}{camera.request_id ? ` · ${camera.request_id}` : ""}</p>
        </article>
        <article className="operation-track" aria-label="Background queue track">
          <h3>Background queue</h3>
          <ol className="pipeline-steps">
            {backgroundStages.map((stage, index) => (
              <li key={stage} className={activeJob?.status === stage ? "current" : stage === "completed" && queue.counts.completed > 0 ? "complete" : "upcoming"} aria-current={activeJob?.status === stage ? "step" : undefined}>
                <span aria-hidden="true">{index + 1}</span><strong>{formatLabel(stage)}</strong>
              </li>
            ))}
          </ol>
          <p className="pipeline-detail">{queue.counts.queued} queued · {queue.counts.processing} active · {queue.counts.failed} failed · {queue.counts.completed} completed</p>
        </article>
      </div>
      <div className="mobile-operation-summary" role="region" aria-label="Mobile operation status">
        <article>
          <h3><span className={`dot ${camera.activity === "capturing" ? "working" : "idle"}`} />Camera</h3>
          <strong>{formatLabel(camera.activity)}</strong>
          <p>{camera.pending_captures} pending capture{camera.pending_captures === 1 ? "" : "s"}</p>
        </article>
        <article>
          <h3><span className={`dot ${(queue.counts.processing + queue.counts.queued) > 0 ? "working" : "ok"}`} />Background clips</h3>
          <strong>{queue.counts.processing} active · {queue.counts.queued} queued</strong>
          <p>{queue.counts.failed} failed · {queue.counts.completed} completed</p>
        </article>
        <article>
          <h3><span className={`dot ${readinessChecking ? "working" : mode === "simulation" || !readiness?.blocking ? "ok" : "attention"}`} />Readiness</h3>
          <strong>{mode === "simulation" ? "Not used" : readinessChecking ? "Checking" : readiness?.blocking ? "Blocked" : readiness ? "Checked" : "Not checked"}</strong>
          <p>{mode === "simulation" ? "Simulation does not check physical equipment." : readinessChecking ? "Previous results are hidden until this check finishes." : "Hardware checks are shown below."}</p>
        </article>
        <details>
          <summary>Show pipeline details</summary>
          <p>Live stage: {formatLabel(run?.stage ?? "idle")} · Background job: {formatLabel(activeJob?.status ?? "none")}</p>
        </details>
      </div>
    </section>
  );
}

function GoProDiagnostics({ report }) {
  if (!report) return null;
  return (
    <section className="gopro-diagnostics" aria-labelledby="gopro-diagnostics-heading" aria-live="polite">
      <header>
        <div>
          <span className="eyebrow">Advanced diagnostics</span>
          <h3 id="gopro-diagnostics-heading">GoPro connection path</h3>
          <p>{report.summary}</p>
        </div>
        <span className={`diagnostic-result ${report.status}`}>{formatLabel(report.status)}</span>
      </header>
      <p className="diagnostic-meta">Target {report.target} · Checked {formatDate(report.checked_at)}</p>
      <ol className="gopro-diagnostic-checks">
        {report.checks.map((check) => (
          <li key={check.check_id} className={check.status}>
            <span className={`check-mark ${check.status}`} aria-hidden="true">
              {check.status === "pass" ? "✓" : check.status === "fail" ? "×" : "?"}
            </span>
            <div>
              <strong>{check.label}</strong>
              <small>{check.evidence}</small>
              {check.status !== "pass" && check.corrective_action && <p>{check.corrective_action}</p>}
            </div>
            <span className={`diagnostic-result ${check.status}`}>{formatLabel(check.status)}</span>
          </li>
        ))}
      </ol>
      <p className="diagnostic-safety">These checks do not start preview or recording and do not change camera settings.</p>
    </section>
  );
}

function ReadinessPanel({
  report, acknowledged, onAcknowledge, onRun, busy, disabled,
  failure, diagnostics, onRunDiagnostics, diagnosticsBusy,
}) {
  const checks = report?.checks ?? [];
  const status = report?.status === "not_checked" ? "not_checked" : report?.blocking ? "blocked" : "ready";
  return (
    <section className="readiness-panel panel" aria-labelledby="readiness-heading">
      <div className="panel-title">
        <div><span className="eyebrow">Before hardware starts</span><h2 id="readiness-heading">Hardware readiness</h2></div>
        <button className="secondary" type="button" onClick={onRun} disabled={busy || disabled || diagnosticsBusy}>
          {busy ? "Checking…" : "Run readiness"}
        </button>
      </div>
      {busy && <p className="panel-empty" role="status">Checking the camera, BearTags and required services. Previous results are hidden until this check finishes.</p>}
      {!busy && status === "not_checked" && !failure && <p className="panel-empty">Readiness has not been checked.</p>}
      {!busy && failure && (
        <section className="readiness-timeout" role="alert" aria-labelledby="readiness-timeout-heading">
          <div>
            <span className="eyebrow">{failure.code === "READINESS_TIMEOUT" ? "Check stopped safely" : "Check failed"}</span>
            <h3 id="readiness-timeout-heading">{failure.code === "READINESS_TIMEOUT" ? "Readiness timed out" : "Readiness failed"}</h3>
            <p>{failure.message}</p>
            {failure.corrective_action && <p>{failure.corrective_action}</p>}
            {failure.code === "READINESS_TIMEOUT" && <small>The full check may be waiting on the camera or another hardware service.</small>}
          </div>
          {failure.code === "READINESS_TIMEOUT" && (
            <button
              className="secondary"
              type="button"
              onClick={onRunDiagnostics}
              disabled={diagnosticsBusy}
            >
              {diagnosticsBusy ? "Running diagnostics…" : "Run advanced GoPro diagnostics"}
            </button>
          )}
        </section>
      )}
      {!busy && checks.length > 0 && [
        ["fail", "Blocking issues"],
        ["warning", "Warnings"],
        ["pass", "Passed"],
      ].map(([groupStatus, groupLabel]) => {
        const group = checks.filter((check) => check.status === groupStatus);
        if (group.length === 0) return null;
        return (
          <section className="readiness-group" key={groupStatus} aria-labelledby={`readiness-${groupStatus}`}>
            <h3 id={`readiness-${groupStatus}`}>{groupLabel} <span>{group.length}</span></h3>
            <ul className="readiness-list">
              {group.map((check) => (
                <li key={check.check_id} className={check.status}>
                  <span className={`check-mark ${check.status}`} aria-hidden="true">
                    {check.status === "pass" ? "✓" : check.status === "warning" ? "!" : "×"}
                  </span>
                  <div>
                    <strong>{check.label}</strong>
                    <small>{check.evidence}</small>
                    {check.status !== "pass" && <p>{check.corrective_action}</p>}
                  </div>
                  {check.check_id === "camera" && check.status === "fail" && (
                    <button
                      className="secondary diagnostic-launch"
                      type="button"
                      onClick={onRunDiagnostics}
                      disabled={busy || diagnosticsBusy}
                    >
                      {diagnosticsBusy ? "Running diagnostics…" : "Run advanced diagnostics"}
                    </button>
                  )}
                  {check.status === "warning" && (
                    <label className="warning-acknowledgement">
                      <input
                        type="checkbox"
                        checked={acknowledged.has(check.check_id)}
                        onChange={(event) => onAcknowledge(check.check_id, event.target.checked)}
                      />
                      I reviewed this warning
                    </label>
                  )}
                </li>
              ))}
            </ul>
          </section>
        );
      })}
      {!busy && <GoProDiagnostics report={diagnostics} />}
    </section>
  );
}

function ReadinessSummary({ report, busy, disabled, onRun, onOpen }) {
  const checks = report?.checks ?? [];
  const passed = checks.filter((check) => check.status === "pass").length;
  const warnings = checks.filter((check) => check.status === "warning").length;
  const failures = checks.filter((check) => check.status === "fail").length;
  const status = busy
    ? { label: "Checking", tone: "working", detail: "Testing hardware and required services" }
    : !report || report.status === "not_checked"
      ? { label: "Not checked", tone: "idle", detail: "Run readiness before hardware starts" }
      : report.status === "failed"
        ? { label: "Check failed", tone: "attention", detail: report.failure?.message ?? "Hardware readiness failed" }
      : report.blocking
        ? { label: "Start blocked", tone: "attention", detail: `${failures} blocking issue${failures === 1 ? "" : "s"}` }
        : warnings > 0
          ? { label: "Review needed", tone: "attention", detail: `${passed} passed / ${warnings} warning${warnings === 1 ? "" : "s"}` }
          : { label: "Ready", tone: "ok", detail: `${passed} check${passed === 1 ? "" : "s"} passed` };
  return (
    <section className="readiness-summary panel" aria-label="Readiness summary" aria-live="polite">
      <div className="readiness-summary-copy">
        <span className="eyebrow">Hardware readiness</span>
        <div className="readiness-summary-state">
          <span className={`dot ${status.tone}`} aria-hidden="true" />
          <strong>{status.label}</strong>
        </div>
        <p>{status.detail}</p>
        {report?.checked_at && <small>Last checked {formatDate(report.checked_at)}</small>}
      </div>
      <div className="readiness-summary-actions">
        <button className="secondary" type="button" onClick={onRun} disabled={busy || disabled}>
          {busy ? "Checking..." : report ? "Run readiness again" : "Run readiness"}
        </button>
        <button className="summary-link" type="button" onClick={onOpen}>View all checks</button>
      </div>
    </section>
  );
}

function FailureCard({ failure, mode, onRetry, retrying }) {
  const presentation = failurePresentation(failure, mode);
  return (
    <article className="failure-card">
      <header>
        <div>
          <span className="failure-stage">{formatLabel(failure.stage)} · {formatLabel(failure.component)}</span>
          <h3>{presentation.headline}</h3>
        </div>
        <time>{formatDate(failure.occurred_at)}</time>
      </header>
      {presentation.impact && <p className="failure-impact">{presentation.impact}</p>}
      <p className="corrective-action">{presentation.correctiveAction}</p>
      <div className="failure-actions">
        {failure.retryable && (
          <button className="primary" type="button" onClick={() => onRetry(failure)} disabled={retrying}>
            {retrying ? "Retrying…" : "Retry operation"}
          </button>
        )}
        <details>
          <summary>Technical details</summary>
          <dl>
            <div><dt>Failure</dt><dd>{failure.failure_id}</dd></div>
            <div><dt>Operation</dt><dd>{failure.operation_id ?? "Not available"}</dd></div>
            {failure.job_id && <div><dt>Clip job</dt><dd>{failure.job_id}</dd></div>}
            <div><dt>Error</dt><dd>{failure.error}</dd></div>
            <div><dt>Attempts</dt><dd>{failure.attempts ?? 1}</dd></div>
          </dl>
        </details>
      </div>
    </article>
  );
}

function RecentRuns({ runs }) {
  return (
    <section className="recent-runs panel" aria-labelledby="recent-runs-heading">
      <div className="panel-title"><div><span className="eyebrow">Restored evidence</span><h2 id="recent-runs-heading">Recent runs</h2></div></div>
      {runs.length === 0 ? <p className="panel-empty">No completed runs yet.</p> : (
        <ol>
          {runs.slice(0, 5).map((run) => {
            const unresolved = run.failures?.filter((failure) => !failure.resolved_at).length ?? 0;
            return (
              <li key={run.run_id}>
                <span className={`run-outcome ${run.stage}`}>{formatLabel(run.stage)}</span>
                <div>
                  <strong>{run.scenario ?? formatLabel(run.mode)}</strong>
                  <small>{formatDate(run.started_at)}</small>
                </div>
                <span>{run.artefacts?.length ?? 0} outputs · {unresolved} failures</span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

function DiagnosticsPage({ events, filteredEvents, minimumLogLevel, onMinimumLogLevelChange }) {
  return (
    <div className="diagnostics-page">
      <div className="detail-page-heading">
        <div>
          <span className="eyebrow">Support tool</span>
          <h2 id="diagnostics-heading">Detailed logs</h2>
          <p>Normal operation does not require this page. Use it to share evidence with support.</p>
        </div>
        <span className="count-badge">{filteredEvents.length}/{events.length} events</span>
      </div>
      <section className="diagnostics panel" aria-label="Detailed logs">
        <div className="diagnostic-controls">
          <p>Raw events are technical evidence. They do not determine the operator status.</p>
          <label className="log-filter">
            <span>Minimum level</span>
            <select aria-label="Minimum log level" value={minimumLogLevel} onChange={onMinimumLogLevelChange}>
              <option value="debug">Debug+</option>
              <option value="info">Info+</option>
              <option value="warning">Warning+</option>
              <option value="error">Error</option>
            </select>
          </label>
        </div>
        <ol className="diagnostic-events">
          {events.length === 0 && <li className="empty"><strong>No diagnostic events yet</strong><small>Evidence appears when a runtime starts.</small></li>}
          {events.length > 0 && filteredEvents.length === 0 && <li className="empty"><strong>No matching events</strong><small>Lower the minimum level to show more.</small></li>}
          {filteredEvents.map((event, index) => (
            <li key={`${event.sequence ?? "event"}-${index}`}>
              <time>{event.at_s == null ? "LIVE" : `T+${Number(event.at_s).toFixed(1)}`}</time>
              <span><strong>{eventMessage(event)}</strong><small>{event.payload?.message ?? event.payload?.error ?? event.payload?.operation_id ?? ""}</small></span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}

function App() {
  const [state, setState] = useState(initialState);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [events, setEvents] = useState([]);
  const [minimumLogLevel, setMinimumLogLevel] = useState("info");
  const [actionNotice, setActionNotice] = useState(null);
  const [copiedSupport, setCopiedSupport] = useState(false);
  const [snapshotReceivedAt, setSnapshotReceivedAt] = useState(null);
  const [playhead, setPlayhead] = useState(0);
  const [capturedClip, setCapturedClip] = useState(null);
  const [displayedMedia, setDisplayedMedia] = useState("scenario");
  const [trackingFrame, setTrackingFrame] = useState(null);
  const [trackingData, setTrackingData] = useState(null);
  const [trackingError, setTrackingError] = useState(null);
  const [trackingRequestVersion, setTrackingRequestVersion] = useState(0);
  const [streamConnected, setStreamConnected] = useState(false);
  const [previewVersion, setPreviewVersion] = useState(0);
  const [previewAvailable, setPreviewAvailable] = useState(false);
  const [acknowledgedWarnings, setAcknowledgedWarnings] = useState(new Set());
  const [goproDiagnostics, setGoproDiagnostics] = useState(null);
  const [busyAction, setBusyAction] = useState("");
  const [requestedStopRunId, setRequestedStopRunId] = useState(null);
  const [stopOutcome, setStopOutcome] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [currentPage, setCurrentPage] = useState(() => pageFromPath(window.location.pathname));
  const videoRef = useRef(null);

  useEffect(() => {
    const handlePopState = () => setCurrentPage(pageFromPath(window.location.pathname));
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function navigateTo(page, event) {
    event?.preventDefault();
    const path = PAGE_PATHS[page];
    if (window.location.pathname !== path) window.history.pushState({}, "", path);
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  async function request(path, options) {
    const response = await fetch(path, options);
    const body = await response.json();
    if (!response.ok) {
      const error = new Error(body.error ?? `HTTP ${response.status}`);
      Object.assign(error, {
        code: body.code,
        correctiveAction: body.corrective_action,
        details: body.details,
      });
      throw error;
    }
    return body;
  }

  function updateSnapshot(next) {
    if (!next) return;
    setSnapshotReceivedAt(new Date().toISOString());
    setState((current) => {
      if (
        Number.isFinite(next.sequence)
        && Number.isFinite(current.sequence)
        && next.sequence < current.sequence
      ) {
        return current;
      }
      return { ...current, ...next };
    });
  }

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    Promise.all([request("/api/health"), request("/api/scenarios")])
      .then(([health, scenarioList]) => {
        updateSnapshot(health);
        setScenarios(scenarioList.scenarios);
        setSelectedScenario(health.scenario ?? preferredScenarioName(scenarioList.scenarios));
      })
      .catch((reason) => setActionNotice(actionFailureNotice("load", reason, {
        mode: initialState.mode,
        streamConnected: false,
      })));

    const source = new EventSource("/api/events");
    source.onopen = () => {
      setStreamConnected(true);
    };
    source.onmessage = ({ data }) => {
      const event = JSON.parse(data);
      if (event.kind === "control_snapshot") {
        updateSnapshot(event.payload);
        return;
      }
      if (event.control_snapshot) updateSnapshot(event.control_snapshot);
      if (event.at_s != null) {
        const nextTime = Number(event.at_s);
        setPlayhead(nextTime);
        if (videoRef.current) {
          if (Math.abs(videoRef.current.currentTime - nextTime) > 0.45) {
            videoRef.current.currentTime = nextTime;
          }
          if (event.kind === "preview_frame") videoRef.current.play().catch(() => {});
        }
      }
      if (event.kind === "capture_completed" && event.payload?.filename) {
        setCapturedClip({
          ...event.payload,
          run_id: event.run_id ?? event.control_snapshot?.active_run?.run_id ?? null,
          scenario: event.control_snapshot?.active_run?.scenario ?? null,
          captured_at: event.emitted_at ?? null,
          url: `/api/captures/${encodeURIComponent(event.payload.filename)}`,
        });
        setDisplayedMedia("capture");
      }
      if (event.kind === "person_detected" && event.payload?.bounding_box) {
        setTrackingFrame({
          detection: {
            bounding_box: event.payload.bounding_box,
            confidence: event.payload.confidence,
          },
          coordinate_space: event.payload.coordinate_space,
        });
      }
      if (event.kind === "tracking_observation") setTrackingFrame(event.payload);
      if (event.kind === "virtual_cameraman_completed") {
        setCapturedClip((current) => current?.run_id && current.run_id !== event.run_id ? current : ({
          ...current,
          ...event.payload,
          run_id: event.run_id ?? current?.run_id ?? null,
          processed_url: `/api/captures/${encodeURIComponent(event.payload.processed_filename)}`,
          debug_url: `/api/captures/${encodeURIComponent(event.payload.debug_video_filename)}`,
          tracking_url: `/api/captures/${encodeURIComponent(event.payload.tracking_filename)}`,
        }));
      }
      if (["runtime_completed", "runtime_failed"].includes(event.kind)) videoRef.current?.pause();
      if (!["preview_frame", "tracking_observation"].includes(event.kind)) {
        setEvents((current) => appendRetainedTraceEvent(current, event));
      }
    };
    source.onerror = () => {
      setStreamConnected(false);
    };
    return () => source.close();
  }, []);

  useEffect(() => {
    const activeRun = state.active_run;
    if (!activeRun) return;
    const restored = restoreCapturedClip(activeRun);
    if (!restored) return;
    setCapturedClip((current) => (
      current?.run_id === restored.run_id && current?.filename === restored.filename
        ? current
        : restored
    ));
  }, [state.active_run]);

  useEffect(() => {
    setActionNotice((current) => reconcileActionNotice(current, state, streamConnected));
  }, [state, streamConnected]);

  useEffect(() => {
    if (!requestedStopRunId) return;
    if (state.active_run?.run_id === requestedStopRunId) return;
    const stoppedRun = state.recent_runs?.find((item) => item.run_id === requestedStopRunId);
    if (stoppedRun?.stage === "stopped") {
      setStopOutcome(stopOutcomePresentation(stoppedRun));
    }
    setRequestedStopRunId(null);
  }, [requestedStopRunId, state.active_run, state.recent_runs]);

  useEffect(() => {
    if (!capturedClip?.tracking_url) {
      setTrackingError(null);
      return undefined;
    }
    let cancelled = false;
    request(capturedClip.tracking_url)
      .then((data) => {
        if (cancelled) return;
        setTrackingData(data);
        setTrackingError(null);
      })
      .catch((reason) => {
        if (!cancelled) setTrackingError(reason);
      });
    return () => { cancelled = true; };
  }, [capturedClip?.tracking_url, trackingRequestVersion]);

  const run = state.active_run;
  const hardwareRunning = state.mode === "hardware" && run?.process_state !== "exited" && Boolean(run);
  useEffect(() => {
    if (!hardwareRunning) {
      setPreviewAvailable(false);
      return undefined;
    }
    setPreviewVersion((current) => current + 1);
    const timer = window.setInterval(() => setPreviewVersion((current) => current + 1), 250);
    return () => window.clearInterval(timer);
  }, [hardwareRunning]);

  const selected = scenarios.find((scenario) => scenario.name === selectedScenario);
  const readinessChecking = state.mode === "hardware" && (
    busyAction === "readiness"
    || busyAction === "start"
    || state.readiness?.status === "checking"
  );
  const connectionState = snapshotReceivedAt == null
    ? "loading"
    : streamConnected ? "connected" : "reconnecting";
  const operator = useMemo(
    () => deriveOperatorView(state, acknowledgedWarnings, {
      connectionState,
      readinessChecking,
      startupGuidance: startupGuidance(state.mode, selected),
      stopRequested: requestedStopRunId === state.active_run?.run_id,
    }),
    [acknowledgedWarnings, connectionState, readinessChecking, requestedStopRunId, selected, state],
  );
  const mediaRun = capturedClip?.run_id
    ? state.active_run?.run_id === capturedClip.run_id
      ? state.active_run
      : state.recent_runs?.find((item) => item.run_id === capturedClip.run_id) ?? null
    : null;
  const mediaIssue = trackingError ? mediaFailureNotice("tracking", trackingError, {
    run: mediaRun,
    filename: capturedClip?.tracking_filename,
  }) : null;
  const filteredEvents = events.filter((event) => showsAtMinimumLogLevel(event, minimumLogLevel));

  function showReadinessTimeout(reason) {
    updateSnapshot({
      mode: "hardware",
      readiness: {
        readiness_schema_version: "1.0",
        status: "failed",
        blocking: true,
        warning_ids: [],
        checks: [],
        failure: {
          code: reason.code,
          message: reason.message,
          corrective_action: reason.correctiveAction ?? null,
          details: reason.details ?? null,
        },
      },
    });
    navigateTo("readiness");
  }

  async function perform(name, action) {
    setBusyAction(name);
    setActionNotice(null);
    setCopiedSupport(false);
    try {
      const next = await action();
      updateSnapshot(next);
      return next;
    } catch (reason) {
      if (reason.code === "READINESS_TIMEOUT") {
        showReadinessTimeout(reason);
      } else {
        setActionNotice(actionFailureNotice(name, reason, {
          run: state.active_run,
          mode: state.mode,
          scenario: selectedScenario,
          streamConnected,
          lastKnownAt: snapshotReceivedAt,
        }));
      }
      return null;
    } finally {
      setBusyAction("");
    }
  }

  function resetMediaContext() {
    setPlayhead(0);
    setCapturedClip(null);
    setTrackingFrame(null);
    setTrackingData(null);
    setTrackingError(null);
    setTrackingRequestVersion(0);
    setDisplayedMedia("scenario");
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  }

  async function chooseMode(mode) {
    setAcknowledgedWarnings(new Set());
    setGoproDiagnostics(null);
    await perform("mode", () => request("/api/mode", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mode }),
    }));
    setEvents([]);
    resetMediaContext();
  }

  async function runReadiness() {
    setAcknowledgedWarnings(new Set());
    setGoproDiagnostics(null);
    setBusyAction("readiness");
    setActionNotice(null);
    try {
      const report = await request("/api/readiness/run", { method: "POST" });
      updateSnapshot({ readiness: report });
      return report;
    } catch (reason) {
      if (reason.code === "READINESS_TIMEOUT") {
        showReadinessTimeout(reason);
      } else {
        setActionNotice(actionFailureNotice("readiness", reason, {
          run: state.active_run,
          mode: state.mode,
          streamConnected,
          lastKnownAt: snapshotReceivedAt,
        }));
      }
      return null;
    } finally {
      setBusyAction("");
    }
  }

  async function runGoproDiagnostics() {
    setBusyAction("gopro-diagnostics");
    setActionNotice(null);
    try {
      const report = await request("/api/readiness/diagnostics/gopro", { method: "POST" });
      setGoproDiagnostics(report);
      return report;
    } catch (reason) {
      setActionNotice(actionFailureNotice("gopro-diagnostics", reason, {
        run: state.active_run,
        mode: state.mode,
        streamConnected,
        lastKnownAt: snapshotReceivedAt,
      }));
      return null;
    } finally {
      setBusyAction("");
    }
  }

  function startRun() {
    setStopOutcome(null);
    resetMediaContext();
    return perform("start", () => request("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        mode: state.mode,
        scenario: selectedScenario,
        acknowledged_warning_ids: [...acknowledgedWarnings],
      }),
    }));
  }

  async function stopRun() {
    const runId = run.run_id;
    setStopOutcome(null);
    setRequestedStopRunId(runId);
    const result = await perform("stop", () => request(`/api/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }));
    if (!result) setRequestedStopRunId(null);
    return result;
  }

  function forceStop() {
    if (!window.confirm("Force stop may leave incomplete artefacts. Continue?")) return;
    perform("force-stop", () => request(`/api/runs/${encodeURIComponent(run.run_id)}/force-stop`, { method: "POST" }));
  }

  function restartRun() {
    return perform("restart", () => request(`/api/runs/${encodeURIComponent(run.run_id)}/restart`, { method: "POST" }));
  }

  function retryFailure(failure) {
    return perform(`retry:${failure.failure_id}`, () => request(
      `/api/runs/${encodeURIComponent(run.run_id)}/failures/${encodeURIComponent(failure.failure_id)}/retry`,
      { method: "POST" },
    ));
  }

  function acknowledgeWarning(checkId, checked) {
    setAcknowledgedWarnings((current) => {
      const next = new Set(current);
      if (checked) next.add(checkId); else next.delete(checkId);
      return next;
    });
  }

  function endFailedRun() {
    if (
      state.mode === "hardware"
      && !window.confirm("Confirm that the camera and runtime are stopped. Ending the failed run keeps its evidence and unlocks setup.")
    ) return;
    return perform("end-failed", () => request(`/api/runs/${encodeURIComponent(run.run_id)}/end`, { method: "POST" }));
  }

  function chooseScenario(scenario) {
    setSelectedScenario(scenario);
    resetMediaContext();
  }

  async function copySupportDetails() {
    if (!actionNotice?.supportDetails) return;
    try {
      await navigator.clipboard.writeText(actionNotice.supportDetails);
      setCopiedSupport(true);
    } catch {
      setCopiedSupport(false);
    }
  }

  function showMedia(kind) {
    setDisplayedMedia(kind);
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  }

  function updateTrackingFromVideo() {
    if (displayedMedia !== "capture" || !trackingData?.frames?.length || !videoRef.current) return;
    const frame = Math.round(videoRef.current.currentTime * trackingData.coordinate_space.fps);
    const nearest = trackingData.frames.reduce(
      (best, item) => Math.abs(item.frame_idx - frame) < Math.abs(best.frame_idx - frame) ? item : best,
      trackingData.frames[0],
    );
    setTrackingFrame({ ...nearest, coordinate_space: trackingData.coordinate_space });
  }

  const mediaUrl = displayedMedia === "capture"
    ? capturedClip?.url
    : displayedMedia === "processed"
      ? capturedClip?.processed_url
      : displayedMedia === "debug"
        ? capturedClip?.debug_url
        : selected?.video_url;
  const mediaTitle = {
    scenario: state.mode === "hardware" ? "Live preview" : "Scenario preview",
    capture: "Extracted clip + live overlay",
    processed: "Virtual cameraman output",
    debug: "Tracking engineering view",
  }[displayedMedia];
  const overlaySpace = trackingFrame?.coordinate_space;
  const detectorBox = trackingFrame?.detection?.bounding_box;
  const estimate = trackingFrame?.estimate;
  const cropBox = trackingFrame?.crop_box;
  const showOverlay = Boolean(
    overlaySpace && (detectorBox || estimate) && ["scenario", "capture"].includes(displayedMedia),
  );
  const phaseTone = operator.summary.tone;
  const readinessIssueCount = state.readiness?.checks?.filter(
    (check) => check.status === "warning" || check.status === "fail",
  ).length ?? 0;
  const missingReadinessWarnings = state.readiness?.warning_ids?.filter(
    (warningId) => !acknowledgedWarnings.has(warningId),
  ) ?? [];
  const pageHeader = {
    overview: {
      title: "Edge Control",
      description: "Operate one nearby Edge node and recover failures safely.",
    },
    readiness: {
      title: "Readiness",
      description: "Check hardware and required services before starting.",
    },
    logs: {
      title: "Logs",
      description: "Technical evidence for support and engineering diagnosis.",
    },
  }[currentPage];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <img className="brand-mark" src={bearVisionLogo} alt="" />
          <span><strong>BearVision</strong><small>Edge Control</small></span>
        </div>
        <nav aria-label="Primary navigation">
          <a href="/" aria-current={currentPage === "overview" ? "page" : undefined} onClick={(event) => navigateTo("overview", event)}>Overview</a>
          <a href="/readiness" aria-current={currentPage === "readiness" ? "page" : undefined} onClick={(event) => navigateTo("readiness", event)}>
            Readiness
            {readinessIssueCount > 0 && <span className="nav-count">{readinessIssueCount}</span>}
          </a>
          <a href="/logs" aria-current={currentPage === "logs" ? "page" : undefined} onClick={(event) => navigateTo("logs", event)}>Logs</a>
        </nav>
        <div className="runtime-state" aria-live="polite">
          <span className={`dot ${phaseTone}`} aria-hidden="true" />
          <span><strong>{operator.summary.label}</strong><small>{formatLabel(state.mode)} runtime</small></span>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div><h1>{pageHeader.title}</h1><p>{pageHeader.description}</p></div>
          <span className={`status-badge ${phaseTone}`}><span className="status-dot" />{operator.summary.label}</span>
        </header>

        {actionNotice && (
          <ActionNotice
            notice={actionNotice}
            copied={copiedSupport}
            onCopy={copySupportDetails}
            onDismiss={() => setActionNotice(null)}
            onRetry={stopRun}
          />
        )}

        {stopOutcome && <StopOutcome outcome={stopOutcome} onDismiss={() => setStopOutcome(null)} />}

        <div className="page">
          {currentPage === "overview" && (
            <>
          <OperatorOverview
            connectionState={connectionState}
            mode={state.mode}
            run={run}
            summary={operator.summary}
          />

          <section className="control-card" id="control" aria-labelledby="control-heading">
            <div className="section-heading">
              <div><span className="eyebrow">Operator setup</span><h2 id="control-heading">Choose how to run</h2></div>
              <p>Configuration is locked while a run remains active.</p>
            </div>
            <div className="controls">
              <fieldset className="mode-group">
                <legend>Runtime mode</legend>
                <div className="segmented-control">
                  <button type="button" aria-pressed={state.mode === "simulation"} className={state.mode === "simulation" ? "selected" : ""} disabled={Boolean(run) || busyAction === "mode"} onClick={() => chooseMode("simulation")}>Simulation</button>
                  <button type="button" aria-pressed={state.mode === "hardware"} className={state.mode === "hardware" ? "selected" : ""} disabled={Boolean(run) || busyAction === "mode"} onClick={() => chooseMode("hardware")}>Hardware</button>
                </div>
              </fieldset>
              {state.mode === "simulation" && (
                <label className="scenario-field">Scenario
                  <select value={selectedScenario} disabled={Boolean(run)} onChange={(event) => chooseScenario(event.target.value)}>
                    {scenarios.map((scenario) => (
                      <option key={scenario.name} value={scenario.name}>
                        {scenario.title ?? scenario.name}{scenario.generated_from ? " · Blender" : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="control-actions">
                {!run && state.mode === "simulation" && (
                  <button className="primary" disabled={!operator.canStart || busyAction === "start" || (state.mode === "simulation" && !selectedScenario)} onClick={startRun}>
                    {busyAction === "start" ? "Starting…" : state.mode === "simulation" ? "Run scenario" : "Start hardware"}
                  </button>
                )}
                {!run && state.mode === "hardware" && readinessChecking && (
                  <button className="primary" disabled>
                    Checking...
                  </button>
                )}
                {!run && state.mode === "hardware" && !readinessChecking && (!state.readiness || state.readiness.status === "not_checked") && (
                  <button className="primary" disabled={busyAction === "mode"} onClick={runReadiness}>Run readiness</button>
                )}
                {!run && state.mode === "hardware" && !readinessChecking && state.readiness?.status === "failed" && (
                  <button className="primary" onClick={(event) => navigateTo("readiness", event)}>Review readiness failure</button>
                )}
                {!run && state.mode === "hardware" && !readinessChecking && state.readiness?.status !== "failed" && state.readiness?.blocking && (
                  <button className="primary" onClick={(event) => navigateTo("readiness", event)}>Review blocking issues</button>
                )}
                {!run && state.mode === "hardware" && !readinessChecking && !state.readiness?.blocking && missingReadinessWarnings.length > 0 && (
                  <button className="primary" onClick={(event) => navigateTo("readiness", event)}>
                    Review readiness warning{missingReadinessWarnings.length === 1 ? "" : "s"}
                  </button>
                )}
                {!run && state.mode === "hardware" && !readinessChecking && state.readiness && !state.readiness.blocking && missingReadinessWarnings.length === 0 && (
                  <button className="primary" disabled={!operator.canStart || busyAction === "start"} onClick={startRun}>
                    {busyAction === "start" ? "Starting..." : "Start hardware"}
                  </button>
                )}
                {operator.canStop && <button className="danger" disabled={busyAction === "stop"} onClick={stopRun}>Stop runtime</button>}
                {operator.canRestart && <button className="primary" disabled={busyAction === "restart"} onClick={restartRun}>Restart runtime</button>}
                {operator.canEndFailedRun && <button className="secondary" disabled={busyAction === "end-failed"} onClick={endFailedRun}>End failed run</button>}
                {operator.canForceStop && <button className="danger force" disabled={busyAction === "force-stop"} onClick={forceStop}>Force stop</button>}
              </div>
            </div>
          </section>

          {state.mode === "hardware" && (
            <ReadinessSummary
              report={state.readiness}
              busy={readinessChecking}
              disabled={Boolean(run) || busyAction === "mode"}
              onRun={runReadiness}
              onOpen={(event) => navigateTo("readiness", event)}
            />
          )}

          <Pipeline mode={state.mode} run={run} readiness={state.readiness} readinessChecking={readinessChecking} summary={operator.summary} now={now} />

          {operator.unresolvedFailures.length > 0 && (
            <section className="failure-section" aria-labelledby="failure-heading" aria-live="assertive">
              <div className="section-heading">
                <div><span className="eyebrow">Action required</span><h2 id="failure-heading">Persistent failures</h2></div>
                <p>Failures remain here until the backend reports them resolved.</p>
              </div>
              {operator.unresolvedFailures.map((failure) => (
                <FailureCard
                  key={failure.failure_id}
                  failure={failure}
                  mode={state.mode}
                  onRetry={retryFailure}
                  retrying={busyAction === `retry:${failure.failure_id}`}
                />
              ))}
            </section>
          )}

          <section className="dashboard" aria-label="Runtime workspace">
            <section className="preview panel" id="preview" aria-labelledby="preview-heading">
              <div className="panel-title">
                <div><span className="eyebrow">Primary work surface</span><h2 id="preview-heading">{mediaTitle}</h2></div>
                <span className="mode-badge">{formatLabel(state.mode)}</span>
              </div>
              {mediaIssue && <MediaIssue issue={mediaIssue} onRetry={() => setTrackingRequestVersion((current) => current + 1)} />}
              <div className="preview-content">
                {hardwareRunning && displayedMedia === "scenario" ? (
                  <div className="video-stage hardware-preview">
                    <img
                      src={`/api/preview/frame.jpg?t=${previewVersion}`}
                      alt="Live GoPro preview"
                      onLoad={() => setPreviewAvailable(true)}
                      onError={() => setPreviewAvailable(false)}
                    />
                    {!previewAvailable && <p>Waiting for the first GoPro frame…</p>}
                  </div>
                ) : mediaUrl ? (
                  <div className="video-stage">
                    <video key={mediaUrl} ref={videoRef} src={mediaUrl} muted playsInline controls onTimeUpdate={updateTrackingFromVideo} />
                    {showOverlay && (
                      <svg className="tracking-overlay" viewBox={`0 0 ${overlaySpace.width_px} ${overlaySpace.height_px}`} preserveAspectRatio="xMidYMid meet" aria-label="Detection and rider position overlay">
                        {detectorBox && (
                          <g className="detector-measurement">
                            <rect x={detectorBox.x_px} y={detectorBox.y_px} width={detectorBox.width_px} height={detectorBox.height_px} />
                            <text x={detectorBox.x_px} y={Math.max(8, detectorBox.y_px - 3)}>YOLO person</text>
                          </g>
                        )}
                        {cropBox && <rect className="crop-window" x={cropBox.x_px} y={cropBox.y_px} width={cropBox.width_px} height={cropBox.height_px} />}
                        {estimate && (
                          <g className="kalman-estimate">
                            <circle cx={estimate.x_px} cy={estimate.y_px} r={trackingFrame.confidence_radius_95_px} />
                            <line x1={estimate.x_px - 6} y1={estimate.y_px} x2={estimate.x_px + 6} y2={estimate.y_px} />
                            <line x1={estimate.x_px} y1={estimate.y_px - 6} x2={estimate.x_px} y2={estimate.y_px + 6} />
                          </g>
                        )}
                      </svg>
                    )}
                    {showOverlay && (
                      <div className="overlay-legend">
                        <span className="green">YOLO person</span>
                        <span className="red">Kalman + RTS · 95 %</span>
                        <span className="cyan">Butterworth camera crop</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                    <div className="reticle" />
                    <strong>{state.mode === "simulation" ? "Behavioural scenario" : "Hardware preview"}</strong>
                    <p>{state.mode === "simulation" ? "This scenario has no recorded video." : "Complete readiness and start hardware to open the GoPro preview."}</p>
                  </>
                )}
                {state.mode === "simulation" && selected && (
                  <div className="sources">
                    {Object.entries(selected.components).map(([component, source]) => <span key={component}>{component}: {source}</span>)}
                    {selected.generated_from && <span>synthetic data: {selected.generated_from.generator}</span>}
                  </div>
                )}
                {capturedClip && (
                  <div className="media-switcher" aria-label="Media view">
                    <button type="button" aria-pressed={displayedMedia === "scenario"} className={displayedMedia === "scenario" ? "selected" : ""} onClick={() => showMedia("scenario")}>{state.mode === "hardware" ? "Live preview" : "Scenario source"}</button>
                    {capturedClip.url && <button type="button" aria-pressed={displayedMedia === "capture"} className={displayedMedia === "capture" ? "selected" : ""} onClick={() => showMedia("capture")}>Extracted clip</button>}
                    {capturedClip.processed_url && <button type="button" aria-pressed={displayedMedia === "processed"} className={displayedMedia === "processed" ? "selected" : ""} onClick={() => showMedia("processed")}>Processed upload</button>}
                    {capturedClip.debug_url && <button type="button" aria-pressed={displayedMedia === "debug"} className={displayedMedia === "debug" ? "selected" : ""} onClick={() => showMedia("debug")}>Tracking view</button>}
                    <small>{displayedMedia === "processed" ? `${capturedClip.processed_filename} · ${formatFileSize(capturedClip.processed_size_bytes)}` : `${capturedClip.filename} · ${formatFileSize(capturedClip.size_bytes)}`}</small>
                  </div>
                )}
                {playhead > 0 && <div className="clock">T+ {playhead.toFixed(1)} s</div>}
              </div>
            </section>

            <aside className="status-rail" id="activity">
              <section className="panel indicators" aria-labelledby="system-heading">
                <div className="panel-title"><div><span className="eyebrow">Supporting information</span><h2 id="system-heading">Operational details</h2></div></div>
                <div className="indicator-list">
                  <Indicator label="BearVision status" status={operator.summary.tone} detail={operator.summary.label} />
                  <Indicator label="Control connection" status={connectionState === "connected" ? "ok" : "attention"} detail={connectionLabel(connectionState)} />
                  <Indicator label="Camera" status={run?.capture_activity?.activity === "capturing" ? "working" : "idle"} detail={`${formatLabel(run?.capture_activity?.activity ?? "idle")} · ${run?.capture_activity?.pending_captures ?? 0} pending`} />
                  <Indicator label="Queue depth" status={(run?.clip_queue?.counts?.queued ?? 0) > 0 ? "working" : "idle"} detail={String(run?.clip_queue?.counts?.queued ?? 0)} />
                  <Indicator label="Current clip job" status={run?.clip_queue?.current_job ? "working" : "idle"} detail={run?.clip_queue?.current_job ?? "None"} />
                  <Indicator label="Oldest queued" status={run?.clip_queue?.oldest_queued_at_utc ? "working" : "idle"} detail={run?.clip_queue?.oldest_queued_at_utc ? formatDate(run.clip_queue.oldest_queued_at_utc) : "None"} />
                  <Indicator label="Failed clips" status={(run?.clip_queue?.counts?.failed ?? 0) > 0 ? "attention" : "ok"} detail={String(run?.clip_queue?.counts?.failed ?? 0)} />
                  <Indicator label="Readiness" status={state.mode === "simulation" ? "idle" : readinessChecking ? "working" : state.readiness?.blocking ? "attention" : state.readiness ? "ok" : "idle"} detail={state.mode === "simulation" ? "Not used in simulation" : readinessChecking ? "Checking" : state.readiness?.blocking ? "Blocked" : state.readiness ? "Checked" : "Not checked"} />
                </div>
              </section>
              <RecentRuns runs={state.recent_runs ?? []} />
            </aside>
          </section>
            </>
          )}

          {currentPage === "readiness" && (
            <section className="readiness-page" aria-labelledby="readiness-page-heading">
              <div className="detail-page-heading">
                <div>
                  <span className="eyebrow">Hardware checks</span>
                  <h2 id="readiness-page-heading">Readiness details</h2>
                  <p>Resolve blocking issues and review warnings before hardware starts.</p>
                </div>
                <button className="secondary" type="button" onClick={(event) => navigateTo("overview", event)}>Back to overview</button>
              </div>
              {state.mode === "hardware" ? (
                <ReadinessPanel
                  report={state.readiness}
                  acknowledged={acknowledgedWarnings}
                  onAcknowledge={acknowledgeWarning}
                  onRun={runReadiness}
                  failure={state.readiness?.failure ?? null}
                  diagnostics={goproDiagnostics}
                  onRunDiagnostics={runGoproDiagnostics}
                  diagnosticsBusy={busyAction === "gopro-diagnostics"}
                  busy={readinessChecking}
                  disabled={Boolean(run) || busyAction === "mode"}
                />
              ) : (
                <section className="readiness-unavailable panel">
                  <h3>Hardware mode is not selected</h3>
                  <p>Simulation does not use physical equipment checks. Return to Overview and select Hardware to run readiness.</p>
                  <button className="secondary" type="button" onClick={(event) => navigateTo("overview", event)}>Open Overview</button>
                </section>
              )}
            </section>
          )}

          {currentPage === "logs" && (
            <DiagnosticsPage
              events={events}
              filteredEvents={filteredEvents}
              minimumLogLevel={minimumLogLevel}
              onMinimumLogLevelChange={(event) => setMinimumLogLevel(event.target.value)}
            />
          )}
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
