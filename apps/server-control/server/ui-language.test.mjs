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
