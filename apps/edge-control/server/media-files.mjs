import { dirname, resolve } from "node:path";

export function safeLeafPath(root, name) {
  const normalizedRoot = resolve(root);
  const candidate = resolve(normalizedRoot, name);
  if (dirname(candidate) !== normalizedRoot) throw new Error("invalid media file name");
  return candidate;
}

export function parseByteRange(range, size) {
  if (!range) return null;
  const match = /^bytes=(\d+)-(\d*)$/.exec(range);
  if (!match) throw new Error("invalid byte range");
  const start = Number(match[1]);
  const end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
  if (start > end || start >= size) throw new Error("byte range is outside media");
  return { start, end };
}
