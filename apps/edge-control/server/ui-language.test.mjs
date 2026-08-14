import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const appRoot = fileURLToPath(new URL("..", import.meta.url));
const uiSource = readFileSync(new URL("../src/main.jsx", import.meta.url), "utf8");
const documentSource = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const styleSource = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");

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
