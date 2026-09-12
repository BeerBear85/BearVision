import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const appRoot = fileURLToPath(new URL("..", import.meta.url));
const uiSource = readFileSync(new URL("../src/main.jsx", import.meta.url), "utf8");
const documentSource = readFileSync(new URL("../index.html", import.meta.url), "utf8");

test("Server Control declares English as its document language", () => {
  assert.match(documentSource, /<html lang="en">/);
});

test("Server Control contains no Danish UI copy", () => {
  assert.doesNotMatch(uiSource, /[æøåÆØÅ]/);
  for (const phrase of [
    "Overblik",
    "Brugere",
    "Opret bruger",
    "Tildel BearTag",
    "Uafklaret",
    "Jobkø",
  ]) {
    assert.equal(uiSource.includes(phrase), false, `${phrase} remains in ${appRoot}`);
  }
});

test("Server Control uses the repository BearVision logo instead of a text badge", () => {
  assert.match(uiSource, /import bearVisionLogo from "\.\.\/\.\.\/\.\.\/logo\/Logo\.svg"/);
  assert.match(uiSource, /<img className="brand-mark" src=\{bearVisionLogo\} alt="" \/>/);
  assert.equal(uiSource.includes('<span className="brand-mark">BV</span>'), false);
});

test("Server Control refreshes queue views with the server summary", () => {
  assert.match(uiSource, /function VideoLibrary\(\{ onError, refreshVersion,/);
  assert.match(uiSource, /\[query, status, page, userFilter, refreshVersion, mutationVersion\]/);
  assert.match(uiSource, /function JobQueue\(\{ onError, refreshVersion \}\)/);
  assert.match(uiSource, /\[status, refreshVersion\]/);
});
test("non-deleting operator actions do not ask for extra confirmation", () => {
  assert.equal(uiSource.includes("window.confirm"), false);
  assert.match(uiSource, /Reassign clip/);
  assert.match(uiSource, /Save and recalculate/);
});
test("Job queue exposes the assigned rider after corrections", () => {
  assert.match(uiSource, /<th>Rider<\/th>/);
  assert.match(uiSource, /job\.displayName \?\? job\.userEmail \?\? "Unassigned"/);
});
