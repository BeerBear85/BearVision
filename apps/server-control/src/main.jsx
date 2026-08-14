import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json", ...options.headers },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? "The request failed");
  return body;
}

const pageCopy = {
  overview: ["Overview", "Operational status and items requiring attention"],
  videos: ["Videos", "Find, browse and verify processed clips"],
  users: ["Users & BearTags", "Manage identities and time-bound assignments"],
  jobs: ["Job queue", "Monitor processing, failures and requeues"],
};
const statusLabels = {
  ready: "Ready", processing: "Processing", processed: "Processed",
  unresolved: "Unresolved", failed: "Failed",
};
const dateFormatter = new Intl.DateTimeFormat("en-GB", {
  dateStyle: "medium", timeStyle: "short",
});

function formatDate(value) {
  return value ? dateFormatter.format(new Date(value)) : "—";
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  return Math.floor(seconds / 60) + ":" + String(Math.round(seconds % 60)).padStart(2, "0");
}

function formatState(value) {
  if (!value) return "Connecting";
  const label = value.replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function Status({ value }) {
  return <span className={"status status-" + value}>{statusLabels[value] ?? value}</span>;
}

function Empty({ children }) {
  return <div className="empty-state">{children}</div>;
}

function Modal({ title, description, onClose, children }) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
    if (event.target === event.currentTarget) onClose();
  }}>
    <section className="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <header><div><h2 id="modal-title">{title}</h2><p>{description}</p></div>
        <button className="icon-button" type="button" onClick={onClose} aria-label="Close">×</button>
      </header>
      {children}
    </section>
  </div>;
}

function VideoLibrary({ onError, refreshVersion, userFilter = "" }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("processed");
  const [page, setPage] = useState(1);
  const [data, setData] = useState({ items: [], page: 1, pageCount: 0, total: 0 });
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          page: String(page), pageSize: "18",
          ...(status ? { status } : {}),
          ...(query ? { query } : {}),
          ...(userFilter ? { userId: userFilter } : {}),
        });
        const result = await api("/api/jobs?" + params);
        setData(result);
        const next = result.items.find((item) => item.jobId === selected) ?? result.items[0];
        setSelected(next?.jobId ?? null);
        onError("");
      } catch (error) { onError(error.message); }
      finally { setLoading(false); }
    }, 220);
    return () => clearTimeout(timer);
  }, [query, status, page, userFilter, refreshVersion]);

  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    api("/api/jobs/" + encodeURIComponent(selected)).then(setDetail)
      .catch((error) => onError(error.message));
  }, [selected, refreshVersion]);

  const candidates = detail?.candidates ?? [];
  return <>
    <div className="toolbar">
      <div className="filter-row">
        <label>Search<input type="search" value={query} onChange={(event) => {
          setQuery(event.target.value); setPage(1);
        }} placeholder="Rider, BearTag or job ID" /></label>
        <label>Status<select value={status} onChange={(event) => {
          setStatus(event.target.value); setPage(1);
        }}>
          <option value="">All</option><option value="processed">Processed</option>
          <option value="unresolved">Unresolved</option><option value="failed">Failed</option>
        </select></label>
      </div>
      <span className="result-count">{data.total} results</span>
    </div>
    <div className="browser-layout">
      <section>
        {loading && <p className="loading">Loading videos…</p>}
        {!loading && data.items.length === 0 && <Empty>No videos match the filters.</Empty>}
        <div className="video-grid">{data.items.map((job) =>
          <button type="button" className={"video-card " + (selected === job.jobId ? "selected" : "")}
            key={job.jobId} onClick={() => setSelected(job.jobId)}
            aria-pressed={selected === job.jobId}>
            <div className="thumbnail">
              {job.video && <img src={"/api/jobs/" + encodeURIComponent(job.jobId) + "/thumbnail"} alt="" loading="lazy" />}
              {job.video && <span className="play-symbol">▶</span>}
              <span className="duration">{formatDuration(job.durationSeconds)}</span>
            </div>
            <div className="video-copy">
              <div className="card-title"><strong>{job.displayName ?? job.userEmail ?? "Unknown rider"}</strong><Status value={job.status} /></div>
              <span>{formatDate(job.captureStartedAt)}</span>
              <span>{job.selectedBearTagId ?? "No BearTag selected"} · {job.jobId}</span>
            </div>
          </button>
        )}</div>
        {data.pageCount > 1 && <div className="pagination">
          <button className="secondary" disabled={page === 1} onClick={() => setPage(page - 1)}>Previous</button>
          <span>Page {data.page} of {data.pageCount}</span>
          <button className="secondary" disabled={page >= data.pageCount} onClick={() => setPage(page + 1)}>Next</button>
        </div>}
      </section>
      <aside className="detail-panel">
        {!detail && <Empty>Select a job to view its details.</Empty>}
        {detail && <>
          {detail.video && <video key={detail.jobId} controls preload="metadata"
            poster={"/api/jobs/" + encodeURIComponent(detail.jobId) + "/thumbnail"}>
            <source src={"/api/jobs/" + encodeURIComponent(detail.jobId) + "/video"}
              type={detail.video?.mimeType ?? "video/mp4"} />
          </video>}
          <div className="detail-heading">
            <div><h2>{detail.displayName ?? detail.userEmail ?? "Unknown rider"}</h2><p>{detail.jobId}</p></div>
            <Status value={detail.status} />
          </div>
          <dl className="facts">
            <dt>Captured</dt><dd>{formatDate(detail.captureStartedAt)}</dd>
            <dt>BearTag</dt><dd>{detail.selectedBearTagId ?? "—"}</dd>
            <dt>User email</dt><dd>{detail.userEmail ?? "—"}</dd>
            <dt>User ID</dt><dd>{detail.selectedUserId ?? "—"}</dd>
            <dt>Assignment</dt><dd>{detail.assignmentId ?? "—"}</dd>
            <dt>Decision</dt><dd>{detail.reason ?? "Awaiting processing"}</dd>
          </dl>
          {candidates.length > 0 && <section className="evidence"><h3>Candidate evidence</h3>
            {candidates.map((candidate) => <div className="score" key={candidate.bearTagId}>
              <span>{candidate.bearTagId}</span>
              <span className="score-track"><span style={{ width: (candidate.combinedScore * 100) + "%" }} /></span>
              <strong>{candidate.combinedScore.toFixed(2)}</strong>
            </div>)}
          </section>}
          <details className="manifest"><summary>Technical manifest</summary><pre>{JSON.stringify(detail.manifest, null, 2)}</pre></details>
        </>}
      </aside>
    </div>
  </>;
}

function UserForm({ onClose, onDone, onError }) {
  const [values, setValues] = useState({ displayName: "", email: "" });
  async function submit(event) {
    event.preventDefault();
    try {
      await api("/api/users", { method: "POST", body: JSON.stringify(values) });
      await onDone(); onClose();
    } catch (error) { onError(error.message); }
  }
  return <form onSubmit={submit}>
    <label>Name<input required autoFocus value={values.displayName} onChange={(e) => setValues({ ...values, displayName: e.target.value })} /></label>
    <label>Email<input required type="email" value={values.email} onChange={(e) => setValues({ ...values, email: e.target.value })} /></label>
    <p className="field-note">A permanent UUID is generated. The email remains an editable contact field.</p>
    <footer className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button className="primary">Create user</button></footer>
  </form>;
}

function TagForm({ onClose, onDone, onError }) {
  const [id, setId] = useState("");
  async function submit(event) {
    event.preventDefault();
    try {
      await api("/api/beartags", { method: "POST", body: JSON.stringify({ id }) });
      await onDone(); onClose();
    } catch (error) { onError(error.message); }
  }
  return <form onSubmit={submit}>
    <label>BearTag ID<input required autoFocus value={id} onChange={(event) => setId(event.target.value)} placeholder="BearTag-812" /></label>
    <footer className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button className="primary">Create BearTag</button></footer>
  </form>;
}

function AssignmentForm({ user, tags, onClose, onDone, onError }) {
  const [values, setValues] = useState({ userId: user.id, bearTagId: "", validFrom: "", validTo: "" });
  const [validation, setValidation] = useState({ state: "idle", message: "" });
  const complete = values.bearTagId && values.validFrom && values.validTo;
  useEffect(() => {
    if (!complete) { setValidation({ state: "idle", message: "" }); return; }
    const timer = setTimeout(async () => {
      setValidation({ state: "checking", message: "Checking the period…" });
      try {
        await api("/api/assignments/validate", { method: "POST", body: JSON.stringify({
          ...values, validFrom: new Date(values.validFrom).toISOString(),
          validTo: new Date(values.validTo).toISOString(),
        }) });
        setValidation({ state: "valid", message: "The period is valid and does not overlap." });
      } catch (error) { setValidation({ state: "invalid", message: error.message }); }
    }, 300);
    return () => clearTimeout(timer);
  }, [values.bearTagId, values.validFrom, values.validTo]);
  async function submit(event) {
    event.preventDefault();
    if (validation.state !== "valid") return;
    try {
      await api("/api/assignments", { method: "POST", body: JSON.stringify({
        ...values, validFrom: new Date(values.validFrom).toISOString(),
        validTo: new Date(values.validTo).toISOString(),
      }) });
      await onDone(); onClose();
    } catch (error) { onError(error.message); }
  }
  return <form onSubmit={submit}>
    <label>User<input value={user.id} readOnly /></label>
    <label>BearTag<select required autoFocus value={values.bearTagId}
      onChange={(e) => setValues({ ...values, bearTagId: e.target.value })}>
      <option value="">Select BearTag</option>{tags.map((tag) => <option key={tag.id}>{tag.id}</option>)}
    </select></label>
    <div className="date-grid">
      <label>Valid from<input required type="datetime-local" value={values.validFrom} onChange={(e) => setValues({ ...values, validFrom: e.target.value })} /></label>
      <label>Valid until<input required type="datetime-local" value={values.validTo} onChange={(e) => setValues({ ...values, validTo: e.target.value })} /></label>
    </div>
    <p className={"validation validation-" + validation.state}>{validation.message || "Times are displayed locally and stored as UTC."}</p>
    <footer className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button className="primary" disabled={validation.state !== "valid"}>Assign BearTag</button></footer>
  </form>;
}

function Users({ onError, onShowVideos }) {
  const [query, setQuery] = useState("");
  const [data, setData] = useState({ items: [], total: 0 });
  const [tags, setTags] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [modal, setModal] = useState(null);
  async function refresh() {
    try {
      const [users, tagData] = await Promise.all([
        api("/api/users?" + new URLSearchParams({ query, pageSize: "100" })),
        api("/api/beartags"),
      ]);
      setData(users); setTags(tagData.items);
      setSelectedId((current) => users.items.some((item) => item.id === current) ? current : users.items[0]?.id ?? null);
      onError("");
    } catch (error) { onError(error.message); }
  }
  useEffect(() => { const timer = setTimeout(refresh, 180); return () => clearTimeout(timer); }, [query]);
  const selected = data.items.find((user) => user.id === selectedId);
  const initials = (name) => name.split(/\s+/).map((part) => part[0]).join("").slice(0, 2).toUpperCase();
  return <>
    <div className="toolbar">
      <div className="filter-row"><label>Search<input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Name, email or BearTag" /></label></div>
      <div className="toolbar-actions"><button className="secondary" onClick={() => setModal("tag")}>Create BearTag</button><button className="primary" onClick={() => setModal("user")}>Create user</button></div>
    </div>
    <div className="users-layout">
      <div className="table-surface"><table>
        <thead><tr><th>User</th><th>Active BearTags</th><th className="numeric">Videos</th></tr></thead>
        <tbody>{data.items.map((user) => <tr key={user.id} className={user.id === selectedId ? "selected-row" : ""} onClick={() => setSelectedId(user.id)}>
          <td><div className="person"><span className="avatar">{initials(user.displayName)}</span><span><strong>{user.displayName}</strong><small>{user.email}</small></span></div></td>
          <td>{user.activeBearTags.join(", ") || "—"}</td><td className="numeric">{user.processedVideoCount}</td>
        </tr>)}</tbody>
      </table></div>
      <aside className="user-detail">
        {!selected && <Empty>No user selected.</Empty>}
        {selected && <>
          <div className="user-heading"><span className="avatar large">{initials(selected.displayName)}</span><div><h2>{selected.displayName}</h2><p>{selected.email}</p><small>{selected.id}</small></div></div>
          <h3>BearTag history</h3>
          {selected.assignments.length === 0 && <p className="muted">No assignments.</p>}
          {selected.assignments.map((item) => <div className={"assignment " + (item.active ? "active" : "")} key={item.id}>
            <strong>{item.bearTagId}{item.active ? " · active" : ""}</strong><span>{formatDate(item.validFrom)} → {formatDate(item.validTo)}</span>
          </div>)}
          <div className="panel-actions"><button className="primary" onClick={() => setModal("assignment")}>Assign BearTag</button><button className="secondary" onClick={() => onShowVideos(selected.id)}>Show videos ({selected.processedVideoCount})</button></div>
        </>}
      </aside>
    </div>
    <section className="tags-section"><div><h2>BearTags</h2><p>{tags.length} registered tags</p></div><div className="tag-list">{tags.map((tag) => <span key={tag.id}>{tag.id}</span>)}</div></section>
    {modal === "user" && <Modal title="Create user" description="The user gets a permanent UUID; email is contact information." onClose={() => setModal(null)}><UserForm onClose={() => setModal(null)} onDone={refresh} onError={onError} /></Modal>}
    {modal === "tag" && <Modal title="Create BearTag" description="The BearTag ID must match the physical device." onClose={() => setModal(null)}><TagForm onClose={() => setModal(null)} onDone={refresh} onError={onError} /></Modal>}
    {modal === "assignment" && selected && <Modal title="Assign BearTag" description="The period is checked against the complete assignment history." onClose={() => setModal(null)}><AssignmentForm user={selected} tags={tags} onClose={() => setModal(null)} onDone={refresh} onError={onError} /></Modal>}
  </>;
}

function JobQueue({ onError, refreshVersion }) {
  const [status, setStatus] = useState("");
  const [data, setData] = useState({ items: [], total: 0 });
  async function refresh() {
    try {
      const params = new URLSearchParams({ pageSize: "100", ...(status ? { status } : {}) });
      setData(await api("/api/jobs?" + params)); onError("");
    } catch (error) { onError(error.message); }
  }
  useEffect(() => { refresh(); }, [status, refreshVersion]);
  async function requeue(jobId) {
    if (!window.confirm("Requeue " + jobId + "?")) return;
    try {
      await api("/api/jobs/" + encodeURIComponent(jobId) + "/requeue", { method: "POST", body: "{}" });
      await refresh();
    } catch (error) { onError(error.message); }
  }
  return <>
    <div className="toolbar"><div className="filter-row"><label>Status<select value={status} onChange={(e) => setStatus(e.target.value)}><option value="">All</option>{Object.entries(statusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label></div><span className="result-count">{data.total} jobs</span></div>
    <div className="table-surface"><table><thead><tr><th>Job</th><th>Status</th><th>Captured</th><th>Decision</th><th /></tr></thead><tbody>{data.items.map((job) => <tr key={job.jobId}><td><strong>{job.jobId}</strong></td><td><Status value={job.status} /></td><td>{formatDate(job.captureStartedAt)}</td><td>{job.reason ?? "—"}</td><td>{["failed", "unresolved"].includes(job.status) && <button className="secondary small" onClick={() => requeue(job.jobId)}>Requeue</button>}</td></tr>)}</tbody></table></div>
  </>;
}

function Overview({ summary, setView }) {
  const counts = summary?.counts ?? {};
  return <>
    <div className="metric-grid">{["ready", "processing", "processed", "unresolved", "failed"].map((status) => <button key={status} className="metric" onClick={() => setView(status === "processed" ? "videos" : "jobs")}><span>{statusLabels[status]}</span><strong>{counts[status] ?? 0}</strong></button>)}</div>
    <section className="attention"><div><h2>Requires attention</h2><p>{summary?.attentionCount ?? 0} unresolved or failed jobs should be reviewed.</p></div><button className="primary" onClick={() => setView("jobs")}>Open job queue</button></section>
  </>;
}

function App() {
  const [view, setView] = useState("videos");
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");
  const [videoUser, setVideoUser] = useState("");
  const [refreshVersion, setRefreshVersion] = useState(0);
  async function refreshSummary() {
    try {
      setSummary(await api("/api/summary"));
      setRefreshVersion((current) => current + 1);
      setError("");
    }
    catch (caught) { setError(caught.message); }
  }
  useEffect(() => {
    refreshSummary(); const timer = setInterval(refreshSummary, 5000);
    return () => clearInterval(timer);
  }, []);
  const workerStatus = summary?.worker?.status ?? "connecting";
  function showUserVideos(userId) { setVideoUser(userId); setView("videos"); }
  function navigate(next) { if (next !== "videos") setVideoUser(""); setView(next); }
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">BV</span><span><strong>BearVision</strong><small>Server Control</small></span></div>
      <nav aria-label="Primary navigation">{[["overview", "Overview"], ["videos", "Videos"], ["users", "Users & BearTags"], ["jobs", "Job queue"]].map(([key, label]) => <button key={key} className={view === key ? "active" : ""} onClick={() => navigate(key)}>{label}</button>)}</nav>
      <div className="worker-state"><span className={"worker-dot " + workerStatus} /><span><strong>Worker: {formatState(workerStatus)}</strong><small>{summary?.worker?.updatedAt ? "Updated " + formatDate(summary.worker.updatedAt) : "Awaiting status"}</small></span></div>
    </aside>
    <main>
      <header className="topbar"><div><h1>{pageCopy[view][0]}</h1><p>{videoUser ? "Filtered by " + videoUser : pageCopy[view][1]}</p></div><button className="secondary" onClick={refreshSummary}>Refresh</button></header>
      {error && <div className="error-banner" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError("")}>×</button></div>}
      <div className="page">
        {view === "overview" && <Overview summary={summary} setView={setView} />}
        {view === "videos" && <VideoLibrary onError={setError} refreshVersion={refreshVersion} userFilter={videoUser} />}
        {view === "users" && <Users onError={setError} onShowVideos={showUserVideos} />}
        {view === "jobs" && <JobQueue onError={setError} refreshVersion={refreshVersion} />}
      </div>
    </main>
  </div>;
}

createRoot(document.getElementById("root")).render(<App />);
