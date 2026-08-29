import assert from "node:assert/strict";
import test from "node:test";
import { adminRequest, appHost, appPort, host, mediaRange } from "./server.mjs";

test("server is restricted to loopback", () => {
  assert.equal(host, "127.0.0.1");
});

test("Android API has a separate LAN listener", () => {
  assert.equal(appHost, "0.0.0.0");
  assert.equal(appPort, 4321);
});

test("job browsing is delegated to paginated Python read models", () => {
  assert.deepEqual(adminRequest("list-jobs", {
    page: "2", pageSize: "18", status: "processed", query: "BearTag-1",
  }), {
    commandSchemaVersion: "1.0", command: "list-jobs",
    page: "2", pageSize: "18", status: "processed", query: "BearTag-1",
  });
});

test("media byte ranges support browser seeking and reject invalid ranges", () => {
  assert.deepEqual(mediaRange("bytes=10-19", 100), {
    start: 10, end: 19, partial: true,
  });
  assert.deepEqual(mediaRange(undefined, 100), {
    start: 0, end: 99, partial: false,
  });
  assert.equal(mediaRange("bytes=100-120", 100), null);
  assert.equal(mediaRange("items=1-2", 100), null);
  assert.deepEqual(mediaRange("bytes=-10", 100), {
    start: 90, end: 99, partial: true,
  });
});

test("assignment mutation is delegated as an exact Python CLI command", () => {
  assert.deepEqual(adminRequest("create-assignment", {
    id: "a-1", userId: "b10e3918-490c-4a3f-859a-e67c12b66680", bearTagId: "tag-1",
    validFrom: "2026-01-01T00:00:00Z", validTo: "2026-01-02T00:00:00Z",
  }), {
    commandSchemaVersion: "1.0", command: "create-assignment",
    id: "a-1", userId: "b10e3918-490c-4a3f-859a-e67c12b66680", bearTagId: "tag-1",
    validFrom: "2026-01-01T00:00:00Z", validTo: "2026-01-02T00:00:00Z",
  });
});

test("email updates retain the UUID identity", () => {
  assert.deepEqual(adminRequest("update-user-email", {
    userId: "b10e3918-490c-4a3f-859a-e67c12b66680",
    email: "new-bear@example.com",
  }), {
    commandSchemaVersion: "1.0", command: "update-user-email",
    userId: "b10e3918-490c-4a3f-859a-e67c12b66680",
    email: "new-bear@example.com",
  });
});

test("user media commands always carry the claimed owner", () => {
  assert.deepEqual(adminRequest("list-user-videos", {
    userId: "bear@example.com", page: "2", pageSize: "12",
  }), {
    commandSchemaVersion: "1.0", command: "list-user-videos",
    userId: "bear@example.com", page: "2", pageSize: "12",
  });
  assert.deepEqual(adminRequest("materialize-user-media", {
    userId: "bear@example.com", jobId: "job-1", kind: "video",
  }), {
    commandSchemaVersion: "1.0", command: "materialize-user-media",
    userId: "bear@example.com", jobId: "job-1", kind: "video",
  });
});
