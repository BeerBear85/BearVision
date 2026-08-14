import { spawn } from "node:child_process";
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
export const appRoot = resolve(here, "..");
export const repoRoot = resolve(appRoot, "..", "..");
const distRoot = join(appRoot, "dist");
const configPath = process.env.BEARVISION_SERVER_CONFIG ?? join(repoRoot, "config", "server.yaml");
export const host = "127.0.0.1";
const port = Number(process.env.BEARVISION_SERVER_CONTROL_PORT ?? 4320);
let worker = null;

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};

export function pythonCommand() {
  if (process.env.BEARVISION_PYTHON) return process.env.BEARVISION_PYTHON;
  const candidate = process.platform === "win32"
    ? join(repoRoot, ".venv", "Scripts", "python.exe")
    : join(repoRoot, ".venv", "bin", "python");
  return existsSync(candidate) ? candidate : "python";
}

export function adminArgs(command, body = {}) {
  if (command === "create-user") return [command, "--email", body.email, "--display-name", body.displayName];
  if (command === "create-tag") return [command, "--id", body.id];
  if (command === "create-assignment" || command === "validate-assignment") return [
    command,
    ...(body.id ? ["--id", body.id] : []),
    "--user-id", body.userId,
    "--bear-tag-id", body.bearTagId,
    "--valid-from", body.validFrom,
    "--valid-to", body.validTo,
  ];
  if (command === "requeue") return [command, "--job-id", body.jobId];
  if (command === "list-jobs") return [
    command,
    "--page", String(body.page ?? 1),
    "--page-size", String(body.pageSize ?? 24),
    ...(body.status ? ["--status", body.status] : []),
    ...(body.query ? ["--query", body.query] : []),
    ...(body.userId ? ["--user-id", body.userId] : []),
  ];
  if (command === "job-detail") return [command, "--job-id", body.jobId];
  if (command === "list-users") return [
    command,
    "--page", String(body.page ?? 1),
    "--page-size", String(body.pageSize ?? 50),
    ...(body.query ? ["--query", body.query] : []),
  ];
  if (command === "materialize-media") return [
    command, "--job-id", body.jobId, "--kind", body.kind,
  ];
  return [command];
}

function runPython(command, body) {
  return new Promise((resolvePromise, reject) => {
    const args = ["-m", "bearvision.server.cli", "--config", configPath, ...adminArgs(command, body)];
    const child = spawn(pythonCommand(), args, { cwd: repoRoot, windowsHide: true });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        try { reject(new Error(JSON.parse(stderr).error)); }
        catch { reject(new Error(stderr.trim() || `Python exited with ${code}`)); }
        return;
      }
      resolvePromise(stdout.trim() ? JSON.parse(stdout) : null);
    });
  });
}

async function readJson(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 64 * 1024) throw new Error("request body is too large");
  }
  return body ? JSON.parse(body) : {};
}

function writeJson(response, status, body) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}

function queryParameters(url) {
  return Object.fromEntries(url.searchParams.entries());
}

export function mediaRange(range, size) {
  if (!range) return { start: 0, end: size - 1, partial: false };
  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match || (!match[1] && !match[2])) return null;
  const suffixLength = !match[1] && match[2] ? Number(match[2]) : null;
  const start = suffixLength === null
    ? Number(match[1])
    : Math.max(0, size - suffixLength);
  const end = suffixLength === null && match[2]
    ? Math.min(Number(match[2]), size - 1)
    : size - 1;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= size) return null;
  return { start, end, partial: true };
}

function serveMedia(request, response, media) {
  const file = resolve(media.path);
  if (!existsSync(file) || !statSync(file).isFile()) {
    writeJson(response, 404, { error: "media not found" });
    return;
  }
  const size = statSync(file).size;
  const selected = mediaRange(request.headers.range, size);
  if (!selected) {
    response.writeHead(416, { "content-range": "bytes */" + size });
    response.end();
    return;
  }
  const headers = {
    "accept-ranges": "bytes",
    "content-type": media.contentType,
    "content-length": selected.end - selected.start + 1,
    "cache-control": "private, max-age=3600",
  };
  if (selected.partial) {
    headers["content-range"] = "bytes " + selected.start + "-" + selected.end + "/" + size;
  }
  response.writeHead(selected.partial ? 206 : 200, headers);
  createReadStream(file, { start: selected.start, end: selected.end }).pipe(response);
}

function serveStatic(request, response) {
  const pathname = new URL(request.url, "http://localhost").pathname;
  const relative = pathname === "/" ? "index.html" : pathname.slice(1);
  const candidate = resolve(distRoot, relative);
  const path = candidate.startsWith(distRoot) && existsSync(candidate) && statSync(candidate).isFile()
    ? candidate : join(distRoot, "index.html");
  response.writeHead(200, { "content-type": mimeTypes[extname(path)] ?? "application/octet-stream" });
  createReadStream(path).pipe(response);
}

async function handle(request, response) {
  const url = new URL(request.url, "http://localhost");
  try {
    if (request.method === "GET" && url.pathname === "/api/snapshot") {
      const snapshot = await runPython("snapshot");
      snapshot.worker.processRunning = Boolean(worker);
      if (!worker) snapshot.worker.status = "stopped";
      writeJson(response, 200, snapshot);
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/summary") {
      const summary = await runPython("summary");
      summary.worker.processRunning = Boolean(worker);
      if (!worker) summary.worker.status = "stopped";
      writeJson(response, 200, summary);
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/jobs") {
      writeJson(response, 200, await runPython("list-jobs", queryParameters(url)));
      return;
    }
    const mediaRoute = url.pathname.match(/^\/api\/jobs\/([A-Za-z0-9._:-]+)\/(video|thumbnail)$/);
    if (request.method === "GET" && mediaRoute) {
      const media = await runPython("materialize-media", {
        jobId: mediaRoute[1],
        kind: mediaRoute[2],
      });
      serveMedia(request, response, media);
      return;
    }
    const jobDetail = url.pathname.match(/^\/api\/jobs\/([A-Za-z0-9._:-]+)$/);
    if (request.method === "GET" && jobDetail) {
      writeJson(response, 200, await runPython("job-detail", { jobId: jobDetail[1] }));
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/users") {
      writeJson(response, 200, await runPython("list-users", queryParameters(url)));
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/beartags") {
      writeJson(response, 200, await runPython("list-tags"));
      return;
    }
    const body = request.method === "POST" ? await readJson(request) : {};
    if (request.method === "POST" && url.pathname === "/api/users") {
      writeJson(response, 201, await runPython("create-user", body)); return;
    }
    if (request.method === "POST" && url.pathname === "/api/beartags") {
      writeJson(response, 201, await runPython("create-tag", body)); return;
    }
    if (request.method === "POST" && url.pathname === "/api/assignments") {
      writeJson(response, 201, await runPython("create-assignment", body)); return;
    }
    if (request.method === "POST" && url.pathname === "/api/assignments/validate") {
      writeJson(response, 200, await runPython("validate-assignment", body)); return;
    }
    const requeue = url.pathname.match(/^\/api\/jobs\/([A-Za-z0-9._:-]+)\/requeue$/);
    if (request.method === "POST" && requeue) {
      writeJson(response, 200, await runPython("requeue", { jobId: requeue[1] })); return;
    }
    if (url.pathname.startsWith("/api/")) { writeJson(response, 404, { error: "not found" }); return; }
    serveStatic(request, response);
  } catch (error) {
    writeJson(response, 400, { error: error.message });
  }
}

function startWorker() {
  const args = ["-m", "bearvision.server.cli", "--config", configPath, "worker"];
  worker = spawn(pythonCommand(), args, { cwd: repoRoot, windowsHide: true, stdio: "inherit" });
  worker.on("exit", () => { worker = null; });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  startWorker();
  const server = createServer(handle);
  server.listen(port, host, () => console.log(`BearVision Server Control: http://${host}:${port}`));
  const stop = () => { if (worker) worker.kill(); server.close(); };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
}
