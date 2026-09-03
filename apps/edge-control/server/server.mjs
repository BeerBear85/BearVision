import { spawn } from "node:child_process";
import { createReadStream, existsSync, readFileSync, readdirSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

import { EventStream } from "./event-stream.mjs";
import { safeLeafPath, parseByteRange } from "./media-files.mjs";
import { ControlError, ReadinessService } from "./readiness-service.mjs";
import { runtimeArguments } from "./runtime-command.mjs";
import { RunState } from "./run-state.mjs";
import { RuntimeSupervisor } from "./runtime-supervisor.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const defaultAppRoot = resolve(here, "..");
const defaultRepoRoot = resolve(defaultAppRoot, "..", "..");

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".jpg": "image/jpeg",
  ".mp4": "video/mp4",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function writeJson(response, status, body) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}

function writeError(response, error) {
  const status = Number.isInteger(error.status) ? error.status : 400;
  writeJson(response, status, {
    code: error.code ?? "INVALID_REQUEST",
    error: error.message,
    corrective_action: error.correctiveAction ?? null,
    details: error.details ?? null,
  });
}

async function readJson(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 64 * 1024) {
      throw new ControlError("REQUEST_TOO_LARGE", "Request body is too large.", { status: 413 });
    }
  }
  try {
    return body ? JSON.parse(body) : {};
  } catch {
    throw new ControlError("INVALID_JSON", "Request body must be valid JSON.");
  }
}

function pythonCommand(repoRoot) {
  if (process.env.BEARVISION_PYTHON) return process.env.BEARVISION_PYTHON;
  const candidates = process.platform === "win32"
    ? [join(repoRoot, ".venv", "Scripts", "python.exe")]
    : [join(repoRoot, ".venv", "bin", "python")];
  return candidates.find(existsSync) ?? "python";
}

function serveMedia(request, response, filePath) {
  const size = statSync(filePath).size;
  const range = request.headers.range;
  if (!range) {
    response.writeHead(200, {
      "accept-ranges": "bytes",
      "content-length": size,
      "content-type": mimeTypes[extname(filePath)] ?? "application/octet-stream",
    });
    createReadStream(filePath).pipe(response);
    return;
  }
  let parsed;
  try {
    parsed = parseByteRange(range, size);
  } catch {
    response.writeHead(416, { "content-range": `bytes */${size}` });
    response.end();
    return;
  }
  response.writeHead(206, {
    "accept-ranges": "bytes",
    "content-length": parsed.end - parsed.start + 1,
    "content-range": `bytes ${parsed.start}-${parsed.end}/${size}`,
    "content-type": mimeTypes[extname(filePath)] ?? "application/octet-stream",
  });
  createReadStream(filePath, parsed).pipe(response);
}

function runJsonProcess(command, args, { cwd, env = process.env } = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, { cwd, env, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      if (stdout.length > 1024 * 1024) child.kill("SIGTERM");
    });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.once("error", rejectPromise);
    child.once("exit", () => {
      try {
        const line = stdout.trim().split(/\r?\n/).at(-1);
        if (!line) throw new Error(stderr.trim() || "Python command returned no JSON");
        resolvePromise(JSON.parse(line));
      } catch {
        rejectPromise(new ControlError(
          "PYTHON_COMMAND_FAILED",
          "The Python readiness command failed.",
          { status: 502, correctiveAction: "Review the Python runtime logs.", details: stderr.trim() },
        ));
      }
    });
  });
}

export function createEdgeControlServer(options = {}) {
  const appRoot = options.appRoot ?? defaultAppRoot;
  const repoRoot = options.repoRoot ?? defaultRepoRoot;
  const distRoot = options.distRoot ?? join(appRoot, "dist");
  const scenarioRoot = options.scenarioRoot ?? join(repoRoot, "specs", "scenarios");
  const configPath = options.configPath ?? process.env.BEARVISION_CONFIG_PATH
    ?? join(repoRoot, "config", "edge.yaml");
  const captureRoot = options.captureRoot ?? process.env.BEARVISION_CAPTURE_ROOT
    ?? join(repoRoot, "temp", "captures");
  const scratchRoot = options.scratchRoot ?? process.env.BEARVISION_SCRATCH_ROOT
    ?? join(repoRoot, "temp", "scratch");
  const localQueueRoot = options.localQueueRoot ?? process.env.BEARVISION_LOCAL_QUEUE_ROOT
    ?? join(repoRoot, "temp", "simulation-queue");
  const previewFramePath = join(scratchRoot, "live-preview.jpg");
  const stateFile = options.persistState === false
    ? null
    : options.stateFile ?? process.env.BEARVISION_CONTROL_STATE_PATH
      ?? join(scratchRoot, "edge-control", "runs.json");
  const state = options.state ?? new RunState({ stateFile });

  function scenarioNames() {
    return readdirSync(scenarioRoot)
      .filter((name) => name.endsWith(".yaml") && statSync(join(scenarioRoot, name)).isFile())
      .sort();
  }

  function safeScenario(name) {
    if (!scenarioNames().includes(name)) {
      throw new ControlError("INVALID_REQUEST", "Unknown scenario.", {
        correctiveAction: "Select a scenario from the current catalog.",
      });
    }
    return join(scenarioRoot, name);
  }

  function scenarioDetails(name) {
    const document = parseYaml(readFileSync(safeScenario(name), "utf8"));
    const videoPath = document.video?.path ?? null;
    return {
      name,
      title: document.title ?? name.replace(/\.yaml$/i, "").replaceAll("-", " "),
      description: document.description ?? null,
      duration_s: document.duration_s ?? null,
      scenario_schema_version: document.scenario_schema_version,
      components: document.components ?? {
        frames: "synthetic", detector: "declared", bear_tag: "synthetic",
        camera: "simulated", storage: "memory",
      },
      video_url: videoPath ? `/api/scenarios/${encodeURIComponent(name)}/video` : null,
      generated_from: document.generated_from ?? null,
    };
  }

  function scenarioVideo(name) {
    const document = parseYaml(readFileSync(safeScenario(name), "utf8"));
    if (!document.video?.path) throw new ControlError("MEDIA_NOT_FOUND", "Scenario has no video.", { status: 404 });
    const candidate = resolve(repoRoot, document.video.path);
    if (!candidate.startsWith(`${repoRoot}\\`) && !candidate.startsWith(`${repoRoot}/`)) {
      throw new ControlError("INVALID_MEDIA_PATH", "Scenario video must stay inside the repository.");
    }
    if (!existsSync(candidate) || !statSync(candidate).isFile()) {
      throw new ControlError("MEDIA_NOT_FOUND", "Scenario video does not exist.", { status: 404 });
    }
    return candidate;
  }

  const runReadiness = options.runReadiness ?? (() => runJsonProcess(
    pythonCommand(repoRoot),
    [
      "-m", "bearvision.control", "preflight", "--config", configPath,
      "--capture-dir", captureRoot, "--scratch-dir", scratchRoot,
    ],
    { cwd: repoRoot },
  ));
  const readiness = options.readiness ?? new ReadinessService({ runCommand: runReadiness });
  const eventStream = options.eventStream ?? new EventStream({ getSnapshot: () => ({
    ...state.snapshot(), readiness: readiness.current(),
  }) });
  const publishControlEvent = (event, stateSnapshot = state.snapshot()) => eventStream.publish({
    ...event,
    ...(stateSnapshot == null ? {} : {
      control_snapshot: { ...stateSnapshot, readiness: readiness.current() },
    }),
  });

  const spawnRuntime = options.spawnRuntime ?? (({ mode, scenario, runId }) => {
    if (mode === "hardware") rmSync(previewFramePath, { force: true });
    const args = runtimeArguments({
      mode,
      runId,
      scenarioPath: mode === "simulation" ? safeScenario(scenario) : null,
      configPath,
      captureRoot,
      scratchRoot,
      localQueueRoot,
    });
    return spawn(pythonCommand(repoRoot), args, {
      cwd: repoRoot,
      env: process.env,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
  });

  const supervisor = options.supervisor ?? new RuntimeSupervisor({
    state,
    spawnRuntime,
    stopTimeoutMs: Number(process.env.BEARVISION_STOP_TIMEOUT_MS ?? 10_000),
    validateStart: async ({ mode, scenario, acknowledgedWarnings }) => {
      if (mode === "simulation") safeScenario(scenario);
      else await readiness.assertReady({ acknowledgedWarnings });
    },
    publish: publishControlEvent,
  });

  function snapshot() {
    return { status: "ok", ...state.snapshot(), readiness: readiness.current() };
  }

  function findRun(runId) {
    const current = state.snapshot();
    if (current.active_run?.run_id === runId) return current.active_run;
    return current.recent_runs.find((run) => run.run_id === runId) ?? null;
  }

  function serveStatic(response, pathname) {
    const requested = pathname === "/" ? "index.html" : pathname.slice(1);
    const candidate = resolve(distRoot, normalize(requested));
    const filePath = candidate.startsWith(distRoot) && existsSync(candidate) && statSync(candidate).isFile()
      ? candidate
      : join(distRoot, "index.html");
    if (!existsSync(filePath)) {
      writeJson(response, 503, { code: "GUI_NOT_BUILT", error: "GUI is not built; run pnpm build" });
      return;
    }
    response.writeHead(200, { "content-type": mimeTypes[extname(filePath)] ?? "application/octet-stream" });
    createReadStream(filePath).pipe(response);
  }

  const server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://localhost");
    try {
      if (request.method === "GET" && url.pathname === "/api/health") {
        writeJson(response, 200, snapshot());
      } else if (request.method === "GET" && url.pathname === "/api/scenarios") {
        writeJson(response, 200, {
          scenario_catalog_version: "1.1",
          scenarios: scenarioNames().map(scenarioDetails),
        });
      } else if (request.method === "GET" && url.pathname === "/api/readiness") {
        writeJson(response, 200, readiness.current() ?? {
          readiness_schema_version: "1.0", status: "not_checked", blocking: true,
          warning_ids: [], checks: [],
        });
      } else if (request.method === "POST" && url.pathname === "/api/readiness/run") {
        const report = await readiness.run();
        publishControlEvent({ kind: "readiness_updated", payload: report });
        writeJson(response, 200, report);
      } else if (request.method === "GET" && url.pathname === "/api/runs/current") {
        writeJson(response, 200, state.snapshot().active_run);
      } else if (request.method === "GET" && url.pathname === "/api/runs") {
        const limit = Math.max(1, Math.min(10, Number(url.searchParams.get("limit") ?? 10)));
        writeJson(response, 200, { runs: state.snapshot().recent_runs.slice(0, limit) });
      } else if (request.method === "GET" && /^\/api\/runs\/[^/]+$/.test(url.pathname)) {
        const run = findRun(decodeURIComponent(url.pathname.split("/")[3]));
        if (!run) throw new ControlError("RUN_NOT_FOUND", "Run was not found.", { status: 404 });
        writeJson(response, 200, run);
      } else if (request.method === "POST" && (url.pathname === "/api/runs" || url.pathname === "/api/run")) {
        const body = await readJson(request);
        const mode = body.mode ?? state.snapshot().mode;
        const run = await supervisor.start({
          mode,
          scenario: body.scenario ?? null,
          acknowledgedWarnings: body.acknowledged_warning_ids ?? [],
        });
        writeJson(response, 202, { ...snapshot(), active_run: run });
      } else if (request.method === "POST" && /^\/api\/runs\/[^/]+\/stop$/.test(url.pathname)) {
        const runId = decodeURIComponent(url.pathname.split("/")[3]);
        supervisor.stop(runId);
        writeJson(response, 202, snapshot());
      } else if (request.method === "POST" && url.pathname === "/api/stop") {
        const runId = state.snapshot().active_run?.run_id;
        if (!runId) throw new ControlError("RUN_NOT_ACTIVE", "No runtime is active.", { status: 409 });
        supervisor.stop(runId);
        writeJson(response, 202, snapshot());
      } else if (request.method === "POST" && /^\/api\/runs\/[^/]+\/force-stop$/.test(url.pathname)) {
        const runId = decodeURIComponent(url.pathname.split("/")[3]);
        supervisor.forceStop(runId);
        writeJson(response, 202, snapshot());
      } else if (request.method === "POST" && /^\/api\/runs\/[^/]+\/restart$/.test(url.pathname)) {
        const runId = decodeURIComponent(url.pathname.split("/")[3]);
        await supervisor.restart(runId);
        writeJson(response, 202, snapshot());
      } else if (
        request.method === "POST"
        && /^\/api\/runs\/[^/]+\/failures\/[^/]+\/retry$/.test(url.pathname)
      ) {
        const [, , , runId, , failureId] = url.pathname.split("/");
        supervisor.retry(decodeURIComponent(runId), decodeURIComponent(failureId));
        writeJson(response, 202, snapshot());
      } else if (request.method === "POST" && url.pathname === "/api/mode") {
        const body = await readJson(request);
        state.selectMode(body.mode);
        let report = readiness.current();
        if (body.mode === "hardware") {
          report = await readiness.run();
          publishControlEvent({ kind: "readiness_updated", payload: report });
        }
        publishControlEvent({ kind: "mode_selected", payload: { mode: body.mode } });
        writeJson(response, 200, { ...snapshot(), readiness: report });
      } else if (request.method === "GET" && url.pathname === "/api/events") {
        eventStream.connect(request, response);
      } else if (
        request.method === "GET"
        && /^\/api\/scenarios\/[^/]+\/video$/.test(url.pathname)
      ) {
        serveMedia(request, response, scenarioVideo(decodeURIComponent(url.pathname.split("/")[3])));
      } else if (request.method === "GET" && /^\/api\/captures\/[^/]+$/.test(url.pathname)) {
        const name = decodeURIComponent(url.pathname.split("/")[3]);
        const filePath = safeLeafPath(captureRoot, name);
        if (!existsSync(filePath) || !statSync(filePath).isFile()) {
          throw new ControlError("MEDIA_NOT_FOUND", "Capture does not exist.", { status: 404 });
        }
        if (extname(filePath).toLowerCase() === ".json" && !request.headers.range) {
          response.writeHead(200, { "content-type": "application/json; charset=utf-8" });
          createReadStream(filePath).pipe(response);
        } else {
          serveMedia(request, response, filePath);
        }
      } else if (request.method === "GET" && url.pathname === "/api/preview/frame.jpg") {
        if (!existsSync(previewFramePath) || !statSync(previewFramePath).isFile()) {
          throw new ControlError("PREVIEW_NOT_READY", "Hardware preview is not ready.", { status: 503 });
        }
        response.writeHead(200, {
          "cache-control": "no-store, max-age=0",
          "content-length": statSync(previewFramePath).size,
          "content-type": "image/jpeg",
        });
        createReadStream(previewFramePath).pipe(response);
      } else if (url.pathname.startsWith("/api/")) {
        throw new ControlError("NOT_FOUND", "Route was not found.", { status: 404 });
      } else {
        serveStatic(response, url.pathname);
      }
    } catch (error) {
      writeError(response, error);
    }
  });

  function close() {
    supervisor.shutdown();
    eventStream.close();
    if (server.listening) server.close();
  }

  return { server, state, supervisor, readiness, eventStream, close };
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : null;
if (invokedPath === fileURLToPath(import.meta.url)) {
  const control = createEdgeControlServer();
  const port = Number(process.env.BEARVISION_CONTROL_PORT ?? 4310);
  const host = process.env.BEARVISION_CONTROL_HOST ?? "0.0.0.0";
  control.server.listen(port, host, () => {
    console.log(`BearVision Edge Control listening on http://${host}:${port}`);
  });
  const shutdown = () => control.close();
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}
