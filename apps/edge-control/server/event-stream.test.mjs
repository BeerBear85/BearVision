import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import { EventStream } from "./event-stream.mjs";

function connection() {
  const request = new EventEmitter();
  request.headers = {};
  const response = {
    headers: null,
    chunks: [],
    writeHead(_status, headers) { this.headers = headers; },
    write(chunk) { this.chunks.push(chunk); },
    end() {},
  };
  return { request, response };
}

test("SSE publishes event ids and replays only events after Last-Event-ID", () => {
  const stream = new EventStream({ replayLimit: 3, setTimer: () => 1, clearTimer: () => {} });
  stream.publish({ kind: "one", payload: {} });
  stream.publish({ kind: "two", payload: {} });
  stream.publish({ kind: "three", payload: {} });

  const { request, response } = connection();
  request.headers["last-event-id"] = "1";
  stream.connect(request, response);

  const output = response.chunks.join("");
  assert.doesNotMatch(output, /"kind":"one"/);
  assert.match(output, /id: 2/);
  assert.match(output, /"kind":"two"/);
  assert.match(output, /id: 3/);
});

test("SSE falls back to an authoritative snapshot when replay history is stale", () => {
  const stream = new EventStream({
    replayLimit: 2,
    getSnapshot: () => ({ phase: "uploading" }),
    setTimer: () => 1,
    clearTimer: () => {},
  });
  stream.publish({ kind: "one", payload: {} });
  stream.publish({ kind: "two", payload: {} });
  stream.publish({ kind: "three", payload: {} });

  const { request, response } = connection();
  request.headers["last-event-id"] = "0";
  stream.connect(request, response);

  assert.match(response.chunks.join(""), /"kind":"control_snapshot"/);
  assert.match(response.chunks.join(""), /"phase":"uploading"/);
});

test("SSE sends a snapshot when the browser event id is ahead after server restart", () => {
  const stream = new EventStream({
    getSnapshot: () => ({ phase: "failed" }),
    setTimer: () => 1,
    clearTimer: () => {},
  });
  const { request, response } = connection();
  request.headers["last-event-id"] = "47";

  stream.connect(request, response);

  assert.match(response.chunks.join(""), /"kind":"control_snapshot"/);
  assert.match(response.chunks.join(""), /"phase":"failed"/);
});

test("SSE heartbeat keeps connected clients alive", () => {
  let heartbeat = null;
  const stream = new EventStream({
    setTimer: (callback) => { heartbeat = callback; return 1; },
    clearTimer: () => {},
  });
  const { request, response } = connection();
  stream.connect(request, response);

  heartbeat();
  assert.match(response.chunks.join(""), /: heartbeat/);
});
