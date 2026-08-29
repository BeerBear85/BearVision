import { spawn } from "node:child_process";
import { createReadStream, existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { parse as parseYaml } from "yaml";
import { ControlState } from "./control-state.mjs";
import { parseByteRange, safeLeafPath } from "./media-files.mjs";
import { parseRuntimeEventLine } from "./runtime-events.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(here, "..");
const repoRoot = resolve(appRoot, "..", "..");
const distRoot = join(appRoot, "dist");
const scenarioRoot = join(repoRoot, "specs", "scenarios");
const configPath = join(repoRoot, "config", "edge.yaml");
const captureRoot = join(repoRoot, "temp", "captures");
const localQueueRoot = process.env.BEARVISION_LOCAL_QUEUE_ROOT
  ?? join(repoRoot, "temp", "simulation-queue");
const port = Number(process.env.BEARVISION_CONTROL_PORT ?? 4310);
const state = new ControlState();
const clients = new Set();
let child = null;

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mp4": "video/mp4",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function pythonCommand() {
  if (process.env.BEARVISION_PYTHON) return process.env.BEARVISION_PYTHON;
  const candidates = process.platform === "win32"
    ? [join(repoRoot, ".venv", "Scripts", "python.exe")]
    : [join(repoRoot, ".venv", "bin", "python")];
  return candidates.find(existsSync) ?? "python";
}

function writeJson(response, status, body) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}

async function readJson(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 64 * 1024) throw new Error("request body is too large");
  }
  return body ? JSON.parse(body) : {};
}

function publish(kind, payload = {}, at_s = null) {
  const event = {
    control_event_version: "1.0",
    sequence: state.sequence + 1,
    emitted_at: new Date().toISOString(),
    at_s,
    kind,
    payload,
  };
  state.record(event);
  const message = `data: ${JSON.stringify(state.lastEvent)}\n\n`;
  for (const response of clients) response.write(message);
}

function scenarios() {
  return readdirSync(scenarioRoot)
    .filter((name) => name.endsWith(".yaml") && statSync(join(scenarioRoot, name)).isFile())
    .sort();
}

function safeScenario(name) {
  if (!scenarios().includes(name)) throw new Error("unknown scenario");
  return join(scenarioRoot, name);
}

function scenarioDetails(name) {
  const document = parseYaml(readFileSync(safeScenario(name), "utf8"));
  const videoPath = document.video?.path ?? null;
  return {
    name,
    scenario_schema_version: document.scenario_schema_version,
    components: document.components ?? {
      frames: "synthetic",
      detector: "declared",
      bear_tag: "synthetic",
      camera: "simulated",
      storage: "memory",
    },
    video_url: videoPath ? `/api/scenarios/${encodeURIComponent(name)}/video` : null,
    generated_from: document.generated_from ?? null,
  };
}

function scenarioVideo(name) {
  const document = parseYaml(readFileSync(safeScenario(name), "utf8"));
  if (!document.video?.path) throw new Error("scenario has no video");
  const candidate = resolve(repoRoot, document.video.path);
  if (!candidate.startsWith(`${repoRoot}\\`) && !candidate.startsWith(`${repoRoot}/`)) {
    throw new Error("scenario video must stay inside the repository");
  }
  if (!existsSync(candidate) || !statSync(candidate).isFile()) {
    throw new Error("scenario video does not exist");
  }
  return candidate;
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
  const { start, end } = parsed;
  response.writeHead(206, {
    "accept-ranges": "bytes",
    "content-length": end - start + 1,
    "content-range": `bytes ${start}-${end}/${size}`,
    "content-type": mimeTypes[extname(filePath)] ?? "application/octet-stream",
  });
  createReadStream(filePath, { start, end }).pipe(response);
}

function attachOutput(stream, source) {
  const lines = createInterface({ input: stream });
  lines.on("line", (line) => {
    if (!line.trim()) return;
    try {
      const parsed = parseRuntimeEventLine(line);
      publish(parsed.kind, parsed.payload, parsed.at_s);
    } catch {
      publish("runtime_log", { source, message: line });
    }
  });
}

function startRuntime(mode, scenarioName = null) {
  state.start({ mode, scenario: scenarioName });
  const args = mode === "simulation"
    ? [
      "-m", "bearvision.control", "simulate", safeScenario(scenarioName),
      "--realtime", "--local-queue-root", localQueueRoot, "--config", configPath,
    ]
    : ["-m", "bearvision.control", "hardware", "--config", configPath];
  child = spawn(pythonCommand(), args, { cwd: repoRoot, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
  attachOutput(child.stdout, "stdout");
  attachOutput(child.stderr, "stderr");
  publish("runtime_started", { mode, scenario: scenarioName, pid: child.pid });
  child.on("error", (error) => publish("runtime_failed", { message: error.message }));
  child.on("exit", (code, signal) => {
    publish(code === 0 ? "runtime_completed" : "runtime_failed", { code, signal });
    state.stop(code === 0 ? "completed" : "failed");
    const stopped = `data: ${JSON.stringify(state.lastEvent)}\n\n`;
    for (const response of clients) response.write(stopped);
    child = null;
  });
}

function stopRuntime() {
  if (!child) throw new Error("no runtime is active");
  publish("stop_requested", { pid: child.pid });
  child.kill("SIGTERM");
}

function serveStatic(request, response) {
  const pathname = new URL(request.url, "http://localhost").pathname;
  const requested = pathname === "/" ? "index.html" : pathname.slice(1);
  const candidate = resolve(distRoot, normalize(requested));
  const filePath = candidate.startsWith(distRoot) && existsSync(candidate) && statSync(candidate).isFile()
    ? candidate
    : join(distRoot, "index.html");
  if (!existsSync(filePath)) {
    writeJson(response, 503, { error: "GUI is not built; run pnpm build" });
    return;
  }
  response.writeHead(200, { "content-type": mimeTypes[extname(filePath)] ?? "application/octet-stream" });
  createReadStream(filePath).pipe(response);
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url, "http://localhost");
  try {
    if (request.method === "GET" && url.pathname === "/api/health") {
      writeJson(response, 200, { status: "ok", ...state.snapshot() });
    } else if (request.method === "GET" && url.pathname === "/api/scenarios") {
      writeJson(response, 200, {
        scenario_catalog_version: "1.0",
        scenarios: scenarios().map(scenarioDetails),
      });
    } else if (
      request.method === "GET"
      && /^\/api\/scenarios\/[^/]+\/video$/.test(url.pathname)
    ) {
      const name = decodeURIComponent(url.pathname.split("/")[3]);
      serveMedia(request, response, scenarioVideo(name));
    } else if (
      request.method === "GET"
      && /^\/api\/captures\/[^/]+$/.test(url.pathname)
    ) {
      const name = decodeURIComponent(url.pathname.split("/")[3]);
      const filePath = safeLeafPath(captureRoot, name);
      if (!existsSync(filePath) || !statSync(filePath).isFile()) {
        throw new Error("capture does not exist");
      }
      serveMedia(request, response, filePath);
    } else if (request.method === "GET" && url.pathname === "/api/events") {
      response.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      response.write(`data: ${JSON.stringify({ kind: "control_snapshot", payload: state.snapshot() })}\n\n`);
      clients.add(response);
      request.on("close", () => clients.delete(response));
    } else if (request.method === "POST" && url.pathname === "/api/mode") {
      const body = await readJson(request);
      state.selectMode(body.mode);
      publish("mode_selected", { mode: body.mode });
      writeJson(response, 200, state.snapshot());
    } else if (request.method === "POST" && url.pathname === "/api/run") {
      const body = await readJson(request);
      if (state.mode === "simulation") startRuntime("simulation", body.scenario);
      else startRuntime("hardware");
      writeJson(response, 202, state.snapshot());
    } else if (request.method === "POST" && url.pathname === "/api/stop") {
      stopRuntime();
      writeJson(response, 202, state.snapshot());
    } else if (url.pathname.startsWith("/api/")) {
      writeJson(response, 404, { error: "not found" });
    } else {
      serveStatic(request, response);
    }
  } catch (error) {
    writeJson(response, 400, { error: error.message });
  }
});

server.listen(port, "0.0.0.0", () => {
  console.log(`BearVision Edge Control listening on http://0.0.0.0:${port}`);
});

function shutdown() {
  if (child) child.kill("SIGTERM");
  server.close(() => process.exit(0));
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
