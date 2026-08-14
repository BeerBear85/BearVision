import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const initialState = { mode: "simulation", phase: "loading", scenario: null, last_event: null };

function eventMessage(event) {
  const labels = {
    mode_selected: "Runtime mode changed",
    runtime_started: "Runtime started",
    runtime_completed: "Scenario completed",
    runtime_failed: "Runtime failed",
    person_detected: "Vision detected a person",
    capture_started: "GoPro capture started",
    server_assignment: "Server assignment completed",
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

function Indicator({ label, active, detail }) {
  return (
    <div className="indicator">
      <span className={`dot ${active ? "active" : ""}`} />
      <span><strong>{label}</strong><small>{detail}</small></span>
    </div>
  );
}

function App() {
  const [state, setState] = useState(initialState);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [events, setEvents] = useState([]);
  const [error, setError] = useState("");
  const [playhead, setPlayhead] = useState(0);
  const [capturedClip, setCapturedClip] = useState(null);
  const [displayedMedia, setDisplayedMedia] = useState("scenario");
  const [trackingFrame, setTrackingFrame] = useState(null);
  const [trackingData, setTrackingData] = useState(null);
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
        setEvents((current) => [event, ...current].slice(0, 200));
      }
      request("/api/health").then(setState).catch(() => {});
    };
    source.onerror = () => setError("Event stream disconnected; reconnecting...");
    return () => source.close();
  }, []);

  const running = state.phase !== "idle";
  const current = events[0];
  const selected = scenarios.find((scenario) => scenario.name === selectedScenario);
  const lastAssignment = useMemo(
    () => events.find((event) => event.kind === "server_assignment"),
    [events],
  );

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

  return (
    <main>
      <header>
        <div><span className="eyebrow">BEARVISION 3</span><h1>Edge Control</h1></div>
        <div className={`phase ${state.phase}`}>{state.phase}</div>
      </header>

      <section className="controls">
        <div className="mode" aria-label="Runtime mode">
          <button className={state.mode === "simulation" ? "selected" : ""} disabled={running} onClick={() => chooseMode("simulation")}>Simulation</button>
          <button className={state.mode === "hardware" ? "selected" : ""} disabled={running} onClick={() => chooseMode("hardware")}>Hardware</button>
        </div>
        {state.mode === "simulation" && (
          <select value={selectedScenario} disabled={running} onChange={(event) => setSelectedScenario(event.target.value)}>
            {scenarios.map((scenario) => (
              <option key={scenario.name} value={scenario.name}>
                {scenario.name}{scenario.generated_from ? " · generated from Blender" : ""}
              </option>
            ))}
          </select>
        )}
        <button className="primary" disabled={running || (state.mode === "simulation" && !selectedScenario)} onClick={run}>
          {state.mode === "simulation" ? "Run scenario" : "Start hardware"}
        </button>
        <button className="danger" disabled={!running} onClick={stop}>Stop</button>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="dashboard">
        <div className="preview panel">
          <div className="panel-title">
            <span>{mediaTitle}</span>
            <small>{state.mode}</small>
          </div>
          <div className="preview-content">
            {state.mode === "simulation" && mediaUrl ? (
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
                <p>{state.mode === "simulation" ? "This scenario has no recorded video." : "Preview transport is the next hardware integration slice."}</p>
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
              <div className="media-switcher">
                <button
                  className={displayedMedia === "scenario" ? "selected" : ""}
                  onClick={() => showMedia("scenario")}
                >Scenario source</button>
                <button
                  className={displayedMedia === "capture" ? "selected" : ""}
                  onClick={() => showMedia("capture")}
                >Extracted clip</button>
                {capturedClip.processed_url && (
                  <button
                    className={displayedMedia === "processed" ? "selected" : ""}
                    onClick={() => showMedia("processed")}
                  >Upload clip</button>
                )}
                {capturedClip.debug_url && (
                  <button
                    className={displayedMedia === "debug" ? "selected" : ""}
                    onClick={() => showMedia("debug")}
                  >Tracking view</button>
                )}
                <small>
                  {displayedMedia === "processed" && capturedClip.processed_filename
                    ? `${capturedClip.processed_filename} · ${Math.round(capturedClip.processed_size_bytes / 1024)} KiB`
                    : `${capturedClip.filename} · ${Number(capturedClip.clip_duration_s).toFixed(1)} s · ${Math.round(capturedClip.size_bytes / 1024)} KiB`}
                </small>
              </div>
            )}
            {(current?.at_s != null || playhead > 0) && <div className="clock">T+ {playhead.toFixed(1)} s</div>}
          </div>
        </div>

        <aside>
          <div className="panel indicators">
            <div className="panel-title">System</div>
            <Indicator label="Control server" active detail="Node.js" />
            <Indicator label="Runtime" active={running} detail={state.mode} />
            <Indicator label="Capture" active={events.some((event) => event.kind === "capture_started") && !events.some((event) => event.kind === "capture_completed")} detail="GoPro" />
            <Indicator label="Rider" active={Boolean(lastAssignment?.payload?.rider_id)} detail={lastAssignment?.payload?.rider_id ?? "not assigned"} />
          </div>

          <div className="panel event-panel">
            <div className="panel-title"><span>Event log</span><small>{events.length}</small></div>
            <ol>
              {events.length === 0 && <li className="empty">Waiting for runtime events</li>}
              {events.map((event, index) => (
                <li key={`${event.sequence}-${index}`}>
                  <time>{event.at_s == null ? "LIVE" : `T+${Number(event.at_s).toFixed(1)}`}</time>
                  <span><strong>{eventMessage(event)}</strong><small>{event.payload?.rider_id ?? event.payload?.message ?? ""}</small></span>
                </li>
              ))}
            </ol>
          </div>
        </aside>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
