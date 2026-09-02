import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  appendRetainedTraceEvent,
  runtimeLogLevel,
  showsAtMinimumLogLevel,
} from "./log-level.js";
import {
  deriveOperatorView,
  pipelineForStage,
  restoreCapturedClip,
} from "./operator-model.js";
import "./styles.css";

const initialState = {
  control_api_version: "2.0",
  mode: "simulation",
  phase: "loading",
  active_run: null,
  recent_runs: [],
  readiness: null,
};

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
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
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

function Pipeline({ run, readiness, now }) {
  const failedStage = run?.failures?.find((failure) => !failure.resolved_at)?.stage ?? null;
  const stage = run?.stage ?? (readiness?.blocking ? "readiness" : "readiness");
  const stages = pipelineForStage(stage, failedStage);
  return (
    <section className="pipeline panel" aria-labelledby="pipeline-heading" aria-live="polite">
      <div className="pipeline-heading">
        <div>
          <span className="eyebrow">Live operation</span>
          <h2 id="pipeline-heading">Pipeline</h2>
        </div>
        <span className={`status-badge ${run?.stage === "failed" ? "attention" : run ? "working" : "ok"}`}>
          <span className="status-dot" />
          {run ? formatLabel(run.stage) : readiness?.blocking ? "Not ready" : "Idle"}
          {run?.stage_started_at && <small>{elapsedSince(run.stage_started_at, now)}</small>}
        </span>
      </div>
      <ol className="pipeline-steps" aria-label="Runtime pipeline">
        {stages.map((item, index) => (
          <li key={item.key} className={item.status} aria-current={item.status === "current" ? "step" : undefined}>
            <span aria-hidden="true">{item.status === "complete" ? "✓" : index + 1}</span>
            <strong>{item.label}</strong>
          </li>
        ))}
      </ol>
      <p className="pipeline-detail">
        {run?.current_operation?.operation_id
          ? `Current operation: ${run.current_operation.operation_id}`
          : run ? `${formatLabel(run.process_state)} process` : "Start a run when readiness is clear."}
      </p>
    </section>
  );
}

function ReadinessPanel({ report, acknowledged, onAcknowledge, onRun, busy }) {
  const checks = report?.checks ?? [];
  const status = report?.status === "not_checked" ? "not_checked" : report?.blocking ? "blocked" : "ready";
  return (
    <section className="readiness-panel panel" aria-labelledby="readiness-heading">
      <div className="panel-title">
        <div><span className="eyebrow">Before hardware starts</span><h2 id="readiness-heading">Hardware readiness</h2></div>
        <button className="secondary" type="button" onClick={onRun} disabled={busy}>
          {busy ? "Checking…" : "Run readiness"}
        </button>
      </div>
      {status === "not_checked" && <p className="panel-empty">Readiness has not been checked.</p>}
      {checks.length > 0 && [
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
    </section>
  );
}

function FailureCard({ failure, onRetry, retrying }) {
  return (
    <article className="failure-card">
      <header>
        <div>
          <span className="failure-stage">{formatLabel(failure.stage)} · {formatLabel(failure.component)}</span>
          <h3>{failure.operator_message ?? "The runtime operation failed."}</h3>
        </div>
        <time>{formatDate(failure.occurred_at)}</time>
      </header>
      <p className="corrective-action">{failure.corrective_action}</p>
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

function App() {
  const [state, setState] = useState(initialState);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [events, setEvents] = useState([]);
  const [minimumLogLevel, setMinimumLogLevel] = useState("info");
  const [requestError, setRequestError] = useState(null);
  const [playhead, setPlayhead] = useState(0);
  const [capturedClip, setCapturedClip] = useState(null);
  const [displayedMedia, setDisplayedMedia] = useState("scenario");
  const [trackingFrame, setTrackingFrame] = useState(null);
  const [trackingData, setTrackingData] = useState(null);
  const [streamConnected, setStreamConnected] = useState(false);
  const [previewVersion, setPreviewVersion] = useState(0);
  const [previewAvailable, setPreviewAvailable] = useState(false);
  const [acknowledgedWarnings, setAcknowledgedWarnings] = useState(new Set());
  const [busyAction, setBusyAction] = useState("");
  const [now, setNow] = useState(Date.now());
  const videoRef = useRef(null);

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
        setState(health);
        setScenarios(scenarioList.scenarios);
        setSelectedScenario(health.scenario ?? scenarioList.scenarios[0]?.name ?? "");
      })
      .catch((reason) => setRequestError(reason));

    const source = new EventSource("/api/events");
    source.onopen = () => {
      setStreamConnected(true);
      setRequestError((current) => current?.code === "STREAM_DISCONNECTED" ? null : current);
    };
    source.onmessage = ({ data }) => {
      const event = JSON.parse(data);
      if (event.kind === "control_snapshot") {
        setState(event.payload);
        return;
      }
      if (event.control_snapshot) setState(event.control_snapshot);
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
        setCapturedClip((current) => ({
          ...current,
          ...event.payload,
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
      const error = new Error("Live updates disconnected. Reconnecting…");
      error.code = "STREAM_DISCONNECTED";
      setRequestError(error);
    };
    return () => source.close();
  }, []);

  useEffect(() => {
    const evidenceRun = state.active_run ?? state.recent_runs?.[0];
    const restored = restoreCapturedClip(evidenceRun);
    if (restored) setCapturedClip((current) => current?.filename === restored.filename ? current : restored);
  }, [state.active_run, state.recent_runs]);

  useEffect(() => {
    if (!capturedClip?.tracking_url) return;
    request(capturedClip.tracking_url)
      .then(setTrackingData)
      .catch((reason) => setRequestError(reason));
  }, [capturedClip?.tracking_url]);

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

  const operator = useMemo(
    () => deriveOperatorView(state, acknowledgedWarnings),
    [state, acknowledgedWarnings],
  );
  const selected = scenarios.find((scenario) => scenario.name === selectedScenario);
  const filteredEvents = events.filter((event) => showsAtMinimumLogLevel(event, minimumLogLevel));

  async function perform(name, action) {
    setBusyAction(name);
    setRequestError(null);
    try {
      updateSnapshot(await action());
    } catch (reason) {
      setRequestError(reason);
    } finally {
      setBusyAction("");
    }
  }

  async function chooseMode(mode) {
    await perform("mode", () => request("/api/mode", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mode }),
    }));
    setAcknowledgedWarnings(new Set());
    setEvents([]);
    setPlayhead(0);
    setCapturedClip(null);
    setTrackingFrame(null);
    setTrackingData(null);
    setDisplayedMedia("scenario");
  }

  function runReadiness() {
    setAcknowledgedWarnings(new Set());
    return perform("readiness", async () => {
      const report = await request("/api/readiness/run", { method: "POST" });
      return { readiness: report };
    });
  }

  function startRun() {
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

  function stopRun() {
    return perform("stop", () => request(`/api/runs/${encodeURIComponent(run.run_id)}/stop`, { method: "POST" }));
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
  const phaseTone = run?.stage === "failed"
    ? "attention"
    : run ? "working" : streamConnected ? "ok" : "attention";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">BV</span>
          <span><strong>BearVision</strong><small>Edge Control</small></span>
        </div>
        <nav aria-label="Page sections">
          <a href="#control">Control</a>
          <a href="#pipeline-heading">Pipeline</a>
          <a href="#preview">Preview</a>
          <a href="#diagnostics">Diagnostics</a>
        </nav>
        <div className="runtime-state" aria-live="polite">
          <span className={`dot ${phaseTone}`} aria-hidden="true" />
          <span><strong>{formatLabel(run?.stage ?? state.phase)}</strong><small>{formatLabel(state.mode)} runtime</small></span>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div><h1>Edge Control</h1><p>Operate one nearby Edge node and recover failures safely.</p></div>
          <span className={`status-badge ${phaseTone}`}><span className="status-dot" />{formatLabel(run?.stage ?? state.phase)}</span>
        </header>

        {requestError && (
          <div className="error-banner" role="alert">
            <span>
              <strong>{requestError.message}</strong>
              {requestError.correctiveAction && <small>{requestError.correctiveAction}</small>}
            </span>
            <button type="button" aria-label="Dismiss error" onClick={() => setRequestError(null)}>×</button>
          </div>
        )}

        <div className="page">
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
                  <select value={selectedScenario} disabled={Boolean(run)} onChange={(event) => setSelectedScenario(event.target.value)}>
                    {scenarios.map((scenario) => (
                      <option key={scenario.name} value={scenario.name}>
                        {scenario.title ?? scenario.name}{scenario.generated_from ? " · Blender" : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="control-actions">
                {!run && (
                  <button className="primary" disabled={!operator.canStart || busyAction === "start" || (state.mode === "simulation" && !selectedScenario)} onClick={startRun}>
                    {busyAction === "start" ? "Starting…" : state.mode === "simulation" ? "Run scenario" : "Start hardware"}
                  </button>
                )}
                {operator.canStop && <button className="danger" disabled={busyAction === "stop"} onClick={stopRun}>Stop runtime</button>}
                {operator.canRestart && <button className="primary" disabled={busyAction === "restart"} onClick={restartRun}>Restart runtime</button>}
                {operator.canForceStop && <button className="danger force" disabled={busyAction === "force-stop"} onClick={forceStop}>Force stop</button>}
              </div>
            </div>
          </section>

          <Pipeline run={run} readiness={state.readiness} now={now} />

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
                  onRetry={retryFailure}
                  retrying={busyAction === `retry:${failure.failure_id}`}
                />
              ))}
            </section>
          )}

          {state.mode === "hardware" && (
            <ReadinessPanel
              report={state.readiness}
              acknowledged={acknowledgedWarnings}
              onAcknowledge={acknowledgeWarning}
              onRun={runReadiness}
              busy={busyAction === "readiness" || Boolean(run)}
            />
          )}

          <section className="dashboard" aria-label="Runtime workspace">
            <section className="preview panel" id="preview" aria-labelledby="preview-heading">
              <div className="panel-title">
                <div><span className="eyebrow">Primary work surface</span><h2 id="preview-heading">{mediaTitle}</h2></div>
                <span className="mode-badge">{formatLabel(state.mode)}</span>
              </div>
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
                <div className="panel-title"><div><span className="eyebrow">At a glance</span><h2 id="system-heading">System</h2></div></div>
                <div className="indicator-list">
                  <Indicator label="Control connection" status={streamConnected ? "ok" : "attention"} detail={streamConnected ? "Live" : "Reconnecting"} />
                  <Indicator label="Runtime process" status={run?.process_state === "running" ? "working" : run?.process_state === "exited" && run?.stage === "failed" ? "attention" : "idle"} detail={formatLabel(run?.process_state ?? "idle")} />
                  <Indicator label="Current stage" status={run?.stage === "failed" ? "attention" : run ? "working" : "idle"} detail={formatLabel(run?.stage ?? "idle")} />
                  <Indicator label="Readiness" status={state.readiness?.blocking ? "attention" : state.readiness ? "ok" : "idle"} detail={state.readiness?.blocking ? "Blocked" : state.readiness ? "Checked" : "Not checked"} />
                </div>
              </section>
              <RecentRuns runs={state.recent_runs ?? []} />
            </aside>
          </section>

          <details className="diagnostics panel" id="diagnostics">
            <summary>
              <span><span className="eyebrow">Technical evidence</span><strong>Diagnostics</strong></span>
              <span className="count-badge">{filteredEvents.length}/{events.length}</span>
            </summary>
            <div className="diagnostic-controls">
              <p>Raw events are intended for support and engineering diagnosis.</p>
              <label className="log-filter">
                <span>Minimum level</span>
                <select aria-label="Minimum log level" value={minimumLogLevel} onChange={(event) => setMinimumLogLevel(event.target.value)}>
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
          </details>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
