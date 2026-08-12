import assert from "node:assert/strict";
import test from "node:test";
import { ControlState } from "./control-state.mjs";

test("mode can only change while idle", () => {
  const state = new ControlState();
  state.selectMode("hardware");
  assert.equal(state.snapshot().mode, "hardware");
  state.start({ mode: "hardware" });
  assert.throws(() => state.selectMode("simulation"));
});

test("runtime lifecycle is exposed as a versioned snapshot", () => {
  const state = new ControlState();
  state.start({ mode: "simulation", scenario: "single-rider-success.yaml" });
  state.record({ kind: "runtime_started", payload: {} });
  assert.deepEqual(
    { version: state.snapshot().control_api_version, phase: state.snapshot().phase },
    { version: "1.0", phase: "running" },
  );
  state.stop();
  assert.equal(state.snapshot().phase, "idle");
});
