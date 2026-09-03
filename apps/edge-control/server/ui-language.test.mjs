import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const appRoot = fileURLToPath(new URL("..", import.meta.url));
const uiSource = readFileSync(new URL("../src/main.jsx", import.meta.url), "utf8");
const documentSource = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const styleSource = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
const serverSource = readFileSync(new URL("./server.mjs", import.meta.url), "utf8");

test("Edge Control declares English as its document language", () => {
  assert.match(documentSource, /<html lang="en">/);
});

test("Edge Control contains no Danish UI copy", () => {
  assert.doesNotMatch(uiSource, /[æøåÆØÅ]/);
  for (const phrase of [
    "Overblik",
    "Brugere",
    "Kamera",
    "Optag",
    "Afventer",
    "Fejl",
  ]) {
    assert.equal(uiSource.includes(phrase), false, `${phrase} remains in ${appRoot}`);
  }
});

test("Edge Control keeps the shared operator shell and accessible states", () => {
  for (const token of [
    'className="app-shell"',
    'className="sidebar"',
    'className="topbar"',
    'role="alert"',
    'aria-live="polite"',
    "aria-pressed={state.mode === \"simulation\"}",
  ]) {
    assert.equal(uiSource.includes(token), true, `${token} is missing from the operator UI`);
  }
  assert.match(styleSource, /@media \(max-width: 620px\)/);
  assert.doesNotMatch(styleSource, /min-width:\s*900px/);
});

test("Edge Control does not present server-owned rider assignment", () => {
  assert.equal(uiSource.includes("server_assignment"), false);
  assert.equal(uiSource.includes('label="Rider"'), false);
  assert.equal(uiSource.includes("selectedUserEmail"), false);
});

test("Edge Control supports production runtime paths", () => {
  for (const variable of [
    "BEARVISION_CONFIG_PATH",
    "BEARVISION_CAPTURE_ROOT",
    "BEARVISION_SCRATCH_ROOT",
  ]) {
    assert.equal(serverSource.includes(variable), true, `${variable} is not configurable`);
  }
  assert.match(serverSource, /"--capture-dir", captureRoot/);
  assert.match(serverSource, /"--scratch-dir", scratchRoot/);
});

test("Edge Control exposes and renders the live hardware preview", () => {
  assert.match(serverSource, /\/api\/preview\/frame\.jpg/);
  assert.match(serverSource, /live-preview\.jpg/);
  assert.match(uiSource, /alt="Live GoPro preview"/);
  assert.doesNotMatch(uiSource, /Preview transport is the next hardware integration slice/);
});

test("Edge Control exposes a minimum log-level filter", () => {
  assert.match(uiSource, /aria-label="Minimum log level"/);
  for (const label of ["Debug+", "Info+", "Warning+", "Error"]) {
    assert.equal(uiSource.includes(label), true, `${label} is missing from the log filter`);
  }
});

test("Edge Control presents operator pipeline, readiness and persistent recovery actions", () => {
  for (const token of [
    'aria-label="Live track"',
    'aria-label="Background queue track"',
    'className="failure-card"',
    'className="readiness-panel panel"',
    'className="diagnostics panel"',
    'className="recent-runs panel"',
    "Run readiness",
    "Retry operation",
    "Restart runtime",
    "Force stop",
  ]) {
    assert.equal(uiSource.includes(token), true, `${token} is missing from the operator UI`);
  }
});

test("Edge Control consumes event snapshots instead of polling health after every event", () => {
  assert.match(uiSource, /event\.control_snapshot/);
  assert.doesNotMatch(uiSource, /request\("\/api\/health"\)\.then\(setState\)/);
});
