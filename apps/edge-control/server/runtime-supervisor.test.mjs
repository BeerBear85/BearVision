import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough, Writable } from "node:stream";
import test from "node:test";

import { RunState } from "./run-state.mjs";
import { RuntimeSupervisor } from "./runtime-supervisor.mjs";

function fakeChild() {
  const child = new EventEmitter();
  child.pid = 4321;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.commands = [];
  child.signals = [];
  child.stdin = new Writable({
    write(chunk, _encoding, callback) {
      child.commands.push(chunk.toString());
      callback();
    },
  });
  child.kill = (signal) => {
    child.signals.push(signal);
    return true;
  };
  return child;
}

function nextTurn() {
  return new Promise((resolve) => setImmediate(resolve));
}

test("start validates before changing run state or spawning a child", async () => {
  const state = new RunState();
  let spawned = false;
  const supervisor = new RuntimeSupervisor({
    state,
    validateStart: async () => { throw new Error("unknown scenario"); },
    spawnRuntime: () => { spawned = true; },
  });

  await assert.rejects(() => supervisor.start({ mode: "simulation", scenario: "missing.yaml" }));
  assert.equal(spawned, false);
  assert.equal(state.snapshot().phase, "idle");
});

test("runtime output updates authoritative state and abnormal exit remains actionable", async () => {
  const state = new RunState();
  const child = fakeChild();
  const published = [];
  const supervisor = new RuntimeSupervisor({
    state,
    spawnRuntime: () => child,
    publish: (event) => published.push(event),
  });

  const run = await supervisor.start({ mode: "hardware" });
  assert.equal(state.snapshot().active_run.process_state, "running");
  child.stdout.write(`${JSON.stringify({
    control_event_version: "1.1",
    run_id: run.run_id,
    emitted_at: "2026-09-03T08:15:00Z",
    kind: "lifecycle_changed",
    at_s: null,
    payload: { stage: "recording", operation_id: "capture-1" },
  })}\n`);
  await nextTurn();
  assert.equal(state.snapshot().active_run.stage, "recording");

  child.stdout.write(`${JSON.stringify({
    control_event_version: "1.1",
    run_id: "run-from-an-older-process",
    emitted_at: "2026-09-03T08:16:00Z",
    kind: "lifecycle_changed",
    at_s: null,
    payload: { stage: "uploading", operation_id: "stale-operation" },
  })}\n`);
  await nextTurn();
  assert.equal(state.snapshot().active_run.stage, "recording");

  child.emit("exit", 1, null);
  assert.equal(state.snapshot().active_run.stage, "failed");
  assert.equal(state.snapshot().active_run.process_state, "exited");
  assert.equal(state.snapshot().active_run.run_id, run.run_id);
  assert.equal(published.at(-1).kind, "runtime_failed");
});

test("graceful stop exposes force stop only after timeout", async () => {
  const state = new RunState();
  const child = fakeChild();
  let timeoutCallback = null;
  const supervisor = new RuntimeSupervisor({
    state,
    spawnRuntime: () => child,
    setTimer: (callback) => { timeoutCallback = callback; return 1; },
    clearTimer: () => {},
  });
  const run = await supervisor.start({ mode: "hardware" });

  supervisor.stop(run.run_id);
  assert.deepEqual(child.signals, ["SIGTERM"]);
  assert.equal(state.snapshot().active_run.stop_state, "graceful_requested");
  assert.throws(() => supervisor.forceStop(run.run_id), /not available/);

  timeoutCallback();
  supervisor.forceStop(run.run_id);
  assert.deepEqual(child.signals, ["SIGTERM", "SIGKILL"]);
});

test("retry sends a command only for a backend-declared retryable failure", async () => {
  const state = new RunState();
  const child = fakeChild();
  const supervisor = new RuntimeSupervisor({ state, spawnRuntime: () => child });
  const run = await supervisor.start({ mode: "hardware" });
  state.record({ kind: "component_failed", payload: {
    failure_id: "failure-upload",
    operation_id: "capture-1:publish",
    stage: "uploading",
    component: "job_queue",
    error: "offline",
    retryable: true,
  } });

  supervisor.retry(run.run_id, "failure-upload");
  assert.deepEqual(JSON.parse(child.commands[0]), {
    command_version: "1.0",
    kind: "retry_failure",
    failure_id: "failure-upload",
  });

  state.record({ kind: "component_failed", payload: {
    failure_id: "failure-camera", component: "camera", error: "offline", retryable: false,
  } });
  assert.throws(() => supervisor.retry(run.run_id, "failure-camera"), /not retryable/);
});

test("successful exit publishes completion once and then an archived snapshot", async () => {
  const state = new RunState();
  const child = fakeChild();
  const published = [];
  const supervisor = new RuntimeSupervisor({
    state,
    spawnRuntime: () => child,
    publish: (event, snapshot) => published.push({ event, snapshot }),
  });
  await supervisor.start({ mode: "hardware" });

  child.emit("exit", 0, null);

  assert.equal(published.filter(({ event }) => event.kind === "runtime_completed").length, 1);
  assert.equal(published.at(-1).event.kind, "run_archived");
  assert.equal(published.at(-1).snapshot.active_run, null);
});

test("a Python terminal failure is not duplicated when the process exits", async () => {
  const state = new RunState();
  const child = fakeChild();
  const supervisor = new RuntimeSupervisor({ state, spawnRuntime: () => child });
  const run = await supervisor.start({ mode: "hardware" });
  child.stdout.write(`${JSON.stringify({
    control_event_version: "1.1",
    run_id: run.run_id,
    emitted_at: "2026-09-03T08:15:00Z",
    kind: "component_failed",
    at_s: null,
    payload: {
      failure_id: "failure-runtime",
      component: "runtime",
      error: "camera disconnected",
      severity: "terminal",
      retryable: false,
    },
  })}\n`);
  await nextTurn();

  child.emit("exit", 1, null);

  assert.equal(state.snapshot().active_run.failures.length, 1);
  assert.equal(state.snapshot().active_run.process_state, "exited");
});

test("a child process error clears the child and leaves restartable state", async () => {
  const state = new RunState();
  const child = fakeChild();
  const supervisor = new RuntimeSupervisor({ state, spawnRuntime: () => child });
  await supervisor.start({ mode: "hardware" });

  child.emit("error", new Error("spawn failed"));

  assert.equal(supervisor.child, null);
  assert.equal(state.snapshot().active_run.stage, "failed");
  assert.equal(state.snapshot().active_run.process_state, "exited");
});

test("restart links the replacement run to the failed run", async () => {
  const state = new RunState();
  const firstChild = fakeChild();
  const secondChild = fakeChild();
  const children = [firstChild, secondChild];
  const supervisor = new RuntimeSupervisor({ state, spawnRuntime: () => children.shift() });
  const failedRun = await supervisor.start({ mode: "hardware" });
  firstChild.emit("exit", 1, null);

  const replacement = await supervisor.restart(failedRun.run_id);

  assert.equal(replacement.restart_of_run_id, failedRun.run_id);
});
