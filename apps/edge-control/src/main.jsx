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
    rider_assignment: "Rider assignment completed",
    capture_completed: "Capture completed",
    clip_uploaded: "Clip uploaded",
    tag_observed: "BearTag observation received",
    component_failed: "Component failure",
    stop_requested: "Stop requested",
    runtime_stopped: "Runtime stopped",
    preview_frame: "Preview frame analysed",
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
      if (event.kind === "runtime_stopped" || event.kind === "runtime_completed") {
        videoRef.current?.pause();
      }
      if (event.kind !== "preview_frame") {
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
    () => events.find((event) => event.kind === "rider_assignment"),
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

  const mediaUrl = displayedMedia === "capture" ? capturedClip?.url : selected?.video_url;

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
            <span>{displayedMedia === "capture" ? "Extracted clip" : "Preview"}</span>
            <small>{state.mode}</small>
          </div>
          <div className="preview-content">
            {state.mode === "simulation" && mediaUrl ? (
              <video key={mediaUrl} ref={videoRef} src={mediaUrl} muted playsInline controls />
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
                <small>
                  {capturedClip.filename} · {Number(capturedClip.clip_duration_s).toFixed(1)} s · {Math.round(capturedClip.size_bytes / 1024)} KiB
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
