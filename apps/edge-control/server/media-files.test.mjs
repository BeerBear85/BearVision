import assert from "node:assert/strict";
import test from "node:test";
import { join } from "node:path";
import { parseByteRange, safeLeafPath } from "./media-files.mjs";

test("capture media paths cannot escape their root", () => {
  const root = join("C:", "bearvision", "captures");
  assert.equal(safeLeafPath(root, "capture-180.mp4"), join(root, "capture-180.mp4"));
  assert.throws(() => safeLeafPath(root, "../edge.yaml"));
  assert.throws(() => safeLeafPath(root, "nested/capture.mp4"));
});

test("HTTP media ranges are bounded", () => {
  assert.deepEqual(parseByteRange("bytes=100-199", 1000), { start: 100, end: 199 });
  assert.deepEqual(parseByteRange("bytes=900-", 1000), { start: 900, end: 999 });
  assert.throws(() => parseByteRange("bytes=1000-", 1000));
});
