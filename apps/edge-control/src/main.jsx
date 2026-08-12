import React, { useEffect, useMemo, useState } from "react";
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
        setSelectedScenario(scenarioList.scenarios[0] ?? "");
      })
      .catch((reason) => setError(reason.message));
    const source = new EventSource("/api/events");
    source.onmessage = ({ data }) => {
      const event = JSON.parse(data);
      if (event.kind === "control_snapshot") {
        setState(event.payload);
        return;
      }
      setEvents((current) => [event, ...current].slice(0, 200));
      request("/api/health").then(setState).catch(() => {});
    };
    source.onerror = () => setError("Event stream disconnected; reconnecting...");
    return () => source.close();
  }, []);

  const running = state.phase !== "idle";
  const current = events[0];
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
            {scenarios.map((scenario) => <option key={scenario}>{scenario}</option>)}
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
          <div className="panel-title"><span>Preview</span><small>{state.mode}</small></div>
          <div className="preview-content">
            <div className="reticle" />
            <strong>{state.mode === "simulation" ? "Behavioural scenario" : "Hardware preview"}</strong>
            <p>{state.mode === "simulation" ? "No video is attached to scenario schema 2.0 yet." : "Preview transport is the next hardware integration slice."}</p>
            {current?.at_s != null && <div className="clock">T+ {Number(current.at_s).toFixed(1)} s</div>}
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
