import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  appendRetainedTraceEvent,
  runtimeLogLevel,
  showsAtMinimumLogLevel,
} from "./log-level.js";
import "./styles.css";

const initialState = { mode: "simulation", phase: "loading", scenario: null, last_event: null };

function formatLabel(value) {
  if (!value) return "Connecting";
  const label = value.replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function formatFileSize(bytes) {
  return Number.isFinite(Number(bytes)) ? `${Math.round(Number(bytes) / 1024)} KiB` : "Size unavailable";
}

function eventMessage(event) {
  const logLevel = runtimeLogLevel(event);
  if (logLevel) return `${formatLabel(logLevel)} log`;

  const labels = {
    mode_selected: "Runtime mode changed",
    runtime_started: "Runtime started",
    runtime_completed: "Scenario completed",
    runtime_failed: "Runtime failed",
    person_detected: "Vision detected a person",
    capture_started: "GoPro capture started",
    capture_completed: "Capture completed",
    clip_uploaded: "Clip uploaded",
    tag_observed: "BearTag observation received",
    component_failed: "Component failure",
    stop_requested: "Stop requested",
    runtime_stopped: "Runtime stopped",
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

function App() {
  const [state, setState] = useState(initialState);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [events, setEvents] = useState([]);
  const [minimumLogLevel, setMinimumLogLevel] = useState("info");
  const [error, setError] = useState("");
  const [playhead, setPlayhead] = useState(0);
  const [capturedClip, setCapturedClip] = useState(null);
  const [displayedMedia, setDisplayedMedia] = useState("scenario");
  const [trackingFrame, setTrackingFrame] = useState(null);
  const [trackingData, setTrackingData] = useState(null);
  const [streamConnected, setStreamConnected] = useState(false);
  const [previewVersion, setPreviewVersion] = useState(0);
  const [previewAvailable, setPreviewAvailable] = useState(false);
  const videoRef = useRef(null);

  async function request(path, options) {
    const response = await fetch(path, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`);
    return body;
  }

  useEffect(() => {
    Promise.all([request("/api/health"), request("/api/scenarios")])
      .then(([health, scenarioList]) => {
        setState(health);
        setScenarios(scenarioList.scenarios);
        setSelectedScenario(scenarioList.scenarios[0]?.name ?? "");
      })
      .catch((reason) => setError(reason.message));
    const source = new EventSource("/api/events");
    source.onopen = () => {
      setStreamConnected(true);
      setError((current) => current === "Event stream disconnected; reconnecting..." ? "" : current);
    };
    source.onmessage = ({ data }) => {
      const event = JSON.parse(data);
      if (event.kind === "control_snapshot") {
        setState(event.payload);
        return;
      }
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
      if (event.kind === "tracking_observation") {
        setTrackingFrame(event.payload);
      }
      if (event.kind === "virtual_cameraman_completed") {
        setCapturedClip((current) => ({
          ...current,
          ...event.payload,
          processed_url: `/api/captures/${encodeURIComponent(event.payload.processed_filename)}`,
          debug_url: `/api/captures/${encodeURIComponent(event.payload.debug_video_filename)}`,
          tracking_url: `/api/captures/${encodeURIComponent(event.payload.tracking_filename)}`,
        }));
        request(`/api/captures/${encodeURIComponent(event.payload.tracking_filename)}`)
          .then(setTrackingData)
          .catch((reason) => setError(reason.message));
      }
      if (event.kind === "runtime_stopped" || event.kind === "runtime_completed") {
        videoRef.current?.pause();
      }
      if (event.kind !== "preview_frame" && event.kind !== "tracking_observation") {
        setEvents((current) => appendRetainedTraceEvent(current, event));
      }
      request("/api/health").then(setState).catch(() => {});
    };
    source.onerror = () => {
      setStreamConnected(false);
      setError("Event stream disconnected; reconnecting...");
    };
    return () => source.close();
  }, []);

  useEffect(() => {
    if (state.mode !== "hardware" || state.phase === "idle") {
      setPreviewAvailable(false);
      return undefined;
    }
    setPreviewVersion((current) => current + 1);
    const timer = window.setInterval(
      () => setPreviewVersion((current) => current + 1),
      250,
    );
    return () => window.clearInterval(timer);
  }, [state.mode, state.phase]);

  const running = state.phase !== "idle";
  const current = events[0];
  const selected = scenarios.find((scenario) => scenario.name === selectedScenario);
  async function chooseMode(mode) {
    try {
      setError("");
      setState(await request("/api/mode", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ mode }),
      }));
      setEvents([]);
      setPlayhead(0);
      setCapturedClip(null);
      setTrackingFrame(null);
      setTrackingData(null);
      setDisplayedMedia("scenario");
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      }
    } catch (reason) { setError(reason.message); }
  }

  async function run() {
    try {
      setError("");
      setEvents([]);
      setTrackingFrame(null);
      setTrackingData(null);
      setState(await request("/api/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ scenario: selectedScenario }),
      }));
    } catch (reason) { setError(reason.message); }
  }

  async function stop() {
    try { setState(await request("/api/stop", { method: "POST" })); }
    catch (reason) { setError(reason.message); }
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
    scenario: "Preview",
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

  const captureRunning = events.some((event) => event.kind === "capture_started")
    && !events.some((event) => event.kind === "capture_completed");
  const filteredEvents = events.filter(
    (event) => showsAtMinimumLogLevel(event, minimumLogLevel),
  );
  const phaseTone = ["failed", "error"].includes(state.phase)
    ? "attention"
    : running ? "working" : "ok";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">BV</span>
          <span><strong>BearVision</strong><small>Edge Control</small></span>
        </div>
        <nav aria-label="Page sections">
          <a href="#control">Control</a>
          <a href="#preview">Live preview</a>
          <a href="#activity">Activity</a>
        </nav>
        <div className="runtime-state" aria-live="polite">
          <span className={`dot ${phaseTone}`} aria-hidden="true" />
          <span><strong>{formatLabel(state.phase)}</strong><small>{formatLabel(state.mode)} runtime</small></span>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div><h1>Edge Control</h1><p>Run capture scenarios and verify the complete Edge pipeline.</p></div>
          <span className={`status-badge ${phaseTone}`}><span className="status-dot" />{formatLabel(state.phase)}</span>
        </header>

        {error && <div className="error-banner" role="alert">
          <span>{error}</span>
          <button type="button" aria-label="Dismiss error" onClick={() => setError("")}>×</button>
        </div>}

        <div className="page">
          <section className="control-card" id="control" aria-labelledby="control-heading">
            <div className="section-heading">
              <div><span className="eyebrow">Operator setup</span><h2 id="control-heading">Choose how to run</h2></div>
              <p>Configuration is locked while the runtime is active.</p>
            </div>
            <div className="controls">
              <fieldset className="mode-group">
                <legend>Runtime mode</legend>
                <div className="segmented-control">
                  <button type="button" aria-pressed={state.mode === "simulation"} className={state.mode === "simulation" ? "selected" : ""} disabled={running} onClick={() => chooseMode("simulation")}>Simulation</button>
                  <button type="button" aria-pressed={state.mode === "hardware"} className={state.mode === "hardware" ? "selected" : ""} disabled={running} onClick={() => chooseMode("hardware")}>Hardware</button>
                </div>
              </fieldset>
              {state.mode === "simulation" && (
                <label className="scenario-field">Scenario
                  <select value={selectedScenario} disabled={running} onChange={(event) => setSelectedScenario(event.target.value)}>
                    {scenarios.map((scenario) => (
                      <option key={scenario.name} value={scenario.name}>
                        {scenario.name}{scenario.generated_from ? " · Blender" : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="control-actions">
                <button className="primary" disabled={running || (state.mode === "simulation" && !selectedScenario)} onClick={run}>
                  {state.mode === "simulation" ? "Run scenario" : "Start hardware"}
                </button>
                <button className="danger" disabled={!running} onClick={stop}>Stop runtime</button>
              </div>
            </div>
          </section>

          <section className="dashboard" aria-label="Runtime workspace">
            <section className="preview panel" id="preview" aria-labelledby="preview-heading">
              <div className="panel-title">
                <div><span className="eyebrow">Media</span><h2 id="preview-heading">{mediaTitle}</h2></div>
                <span className="mode-badge">{formatLabel(state.mode)}</span>
              </div>
              <div className="preview-content">
            {state.mode === "hardware" && running ? (
              <div className="video-stage hardware-preview">
                <img
                  src={`/api/preview/frame.jpg?t=${previewVersion}`}
                  alt="Live GoPro preview"
                  onLoad={() => setPreviewAvailable(true)}
                  onError={() => setPreviewAvailable(false)}
                />
                {!previewAvailable && <p>Waiting for the first GoPro frame…</p>}
              </div>
            ) : state.mode === "simulation" && mediaUrl ? (
              <div className="video-stage">
                <video
                  key={mediaUrl}
                  ref={videoRef}
                  src={mediaUrl}
                  muted
                  playsInline
                  controls
                  onTimeUpdate={updateTrackingFromVideo}
                />
                {showOverlay && (
                  <svg
                    className="tracking-overlay"
                    viewBox={`0 0 ${overlaySpace.width_px} ${overlaySpace.height_px}`}
                    preserveAspectRatio="xMidYMid meet"
                    aria-label="Detection and rider position overlay"
                  >
                    {detectorBox && (
                      <g className="detector-measurement">
                        <rect
                          x={detectorBox.x_px}
                          y={detectorBox.y_px}
                          width={detectorBox.width_px}
                          height={detectorBox.height_px}
                        />
                        <text x={detectorBox.x_px} y={Math.max(8, detectorBox.y_px - 3)}>
                          YOLO person
                        </text>
                      </g>
                    )}
                    {cropBox && (
                      <rect
                        className="crop-window"
                        x={cropBox.x_px}
                        y={cropBox.y_px}
                        width={cropBox.width_px}
                        height={cropBox.height_px}
                      />
                    )}
                    {estimate && (
                      <g className="kalman-estimate">
                        <circle
                          cx={estimate.x_px}
                          cy={estimate.y_px}
                          r={trackingFrame.confidence_radius_95_px}
                        />
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
                <p>{state.mode === "simulation" ? "This scenario has no recorded video." : "Start hardware to open the GoPro preview."}</p>
              </>
            )}
            {state.mode === "simulation" && selected && (
              <div className="sources">
                {Object.entries(selected.components).map(([component, source]) => (
                  <span key={component}>{component}: {source}</span>
                ))}
                {selected.generated_from && (
                  <span>synthetic data: {selected.generated_from.generator}</span>
                )}
              </div>
            )}
            {capturedClip && (
              <div className="media-switcher" aria-label="Media view">
                <button
                  type="button"
                  aria-pressed={displayedMedia === "scenario"}
                  className={displayedMedia === "scenario" ? "selected" : ""}
                  onClick={() => showMedia("scenario")}
                >Scenario source</button>
                <button
                  type="button"
                  aria-pressed={displayedMedia === "capture"}
                  className={displayedMedia === "capture" ? "selected" : ""}
                  onClick={() => showMedia("capture")}
                >Extracted clip</button>
                {capturedClip.processed_url && (
                  <button
                    type="button"
                    aria-pressed={displayedMedia === "processed"}
                    className={displayedMedia === "processed" ? "selected" : ""}
                    onClick={() => showMedia("processed")}
                  >Processed upload</button>
                )}
                {capturedClip.debug_url && (
                  <button
                    type="button"
                    aria-pressed={displayedMedia === "debug"}
                    className={displayedMedia === "debug" ? "selected" : ""}
                    onClick={() => showMedia("debug")}
                  >Tracking view</button>
                )}
                <small>
                  {displayedMedia === "processed" && capturedClip.processed_filename
                    ? `${capturedClip.processed_filename} · ${formatFileSize(capturedClip.processed_size_bytes)}`
                    : `${capturedClip.filename} · ${Number(capturedClip.clip_duration_s).toFixed(1)} s · ${formatFileSize(capturedClip.size_bytes)}`}
                </small>
              </div>
            )}
            {(current?.at_s != null || playhead > 0) && <div className="clock">T+ {playhead.toFixed(1)} s</div>}
              </div>
            </section>

            <aside className="status-rail" id="activity">
              <section className="panel indicators" aria-labelledby="system-heading">
                <div className="panel-title"><div><span className="eyebrow">Health</span><h2 id="system-heading">System</h2></div></div>
                <div className="indicator-list">
                  <Indicator label="Control server" status={streamConnected ? "ok" : "attention"} detail={streamConnected ? "Connected" : "Reconnecting"} />
                  <Indicator label="Runtime" status={running ? "working" : "idle"} detail={formatLabel(state.mode)} />
                  <Indicator label="Capture" status={captureRunning ? "working" : "idle"} detail="GoPro" />
                </div>
              </section>

              <section className="panel event-panel" aria-labelledby="activity-heading" aria-live="polite">
                <div className="panel-title">
                  <div><span className="eyebrow">Live trace</span><h2 id="activity-heading">Activity</h2></div>
                  <div className="trace-controls">
                    <label className="log-filter">
                      <span>Minimum level</span>
                      <select
                        aria-label="Minimum log level"
                        value={minimumLogLevel}
                        onChange={(event) => setMinimumLogLevel(event.target.value)}
                      >
                        <option value="debug">Debug+</option>
                        <option value="info">Info+</option>
                        <option value="warning">Warning+</option>
                        <option value="error">Error</option>
                      </select>
                    </label>
                    <span className="count-badge" title={`${filteredEvents.length} shown out of ${events.length} retained`}>
                      {filteredEvents.length}/{events.length}
                    </span>
                  </div>
                </div>
                <ol>
                  {events.length === 0 && <li className="empty"><strong>No activity yet</strong><small>Events will appear when the runtime starts.</small></li>}
                  {events.length > 0 && filteredEvents.length === 0 && (
                    <li className="empty"><strong>No matching activity</strong><small>Lower the minimum level to show more logs.</small></li>
                  )}
                  {filteredEvents.map((event, index) => (
                    <li key={`${event.sequence}-${index}`}>
                      <time>{event.at_s == null ? "LIVE" : `T+${Number(event.at_s).toFixed(1)}`}</time>
                      <span><strong>{eventMessage(event)}</strong><small>{event.payload?.rider_id ?? event.payload?.message ?? ""}</small></span>
                    </li>
                  ))}
                </ol>
              </section>
            </aside>
          </section>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
