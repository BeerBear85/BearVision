import assert from "node:assert/strict";
import test from "node:test";

import { runtimeArguments } from "./runtime-command.mjs";

test("runtime command carries the authoritative run id into Python", () => {
  const common = {
    runId: "run-edge-17",
    configPath: "config/edge.yaml",
    captureRoot: "temp/captures",
    scratchRoot: "temp/scratch",
    localQueueRoot: "temp/simulation-queue",
  };

  assert.deepEqual(runtimeArguments({
    ...common,
    mode: "simulation",
    scenarioPath: "specs/scenarios/operator.yaml",
  }), [
    "-m", "bearvision.control", "simulate", "specs/scenarios/operator.yaml",
    "--run-id", "run-edge-17", "--realtime",
    "--local-queue-root", "temp/simulation-queue",
    "--config", "config/edge.yaml",
  ]);
  assert.deepEqual(runtimeArguments({ ...common, mode: "hardware" }), [
    "-m", "bearvision.control", "hardware",
    "--config", "config/edge.yaml", "--run-id", "run-edge-17",
    "--capture-dir", "temp/captures", "--scratch-dir", "temp/scratch",
  ]);
});
