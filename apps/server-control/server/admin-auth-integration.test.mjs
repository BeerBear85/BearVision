import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  createAdminHandler, handleApp, pythonEnvironment, staticContentType,
} from "./server.mjs";

const adminEnvironment = {
  BEARVISION_ADMIN_USERNAME: "admin",
  BEARVISION_ADMIN_PASSWORD: "test-password",
};

async function listen(handler) {
  const server = createServer(handler);
  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  return server;
}

test("Server Control serves SVG assets with an image MIME type", () => {
  assert.equal(staticContentType("Logo.svg"), "image/svg+xml");
});

test("the real admin handler authenticates UI, API, and media routes before routing", async (context) => {
  const server = await listen(createAdminHandler(adminEnvironment));
  context.after(() => server.close());
  const { port } = server.address();

  for (const pathname of ["/", "/api/summary", "/api/jobs/job-1/video"]) {
    const response = await fetch(`http://127.0.0.1:${port}${pathname}`);
    assert.equal(response.status, 401, pathname);
    assert.equal(
      response.headers.get("www-authenticate"),
      'Basic realm="BearVision Server Control", charset="UTF-8"',
      pathname,
    );
    assert.equal(response.headers.get("cache-control"), "no-store", pathname);
    assert.deepEqual(await response.json(), { error: "authentication required" }, pathname);
  }
});

test("the Android health API remains available without admin credentials", async (context) => {
  const server = await listen(handleApp);
  context.after(() => server.close());
  const { port } = server.address();

  const response = await fetch(`http://127.0.0.1:${port}/api/app/health`);

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { status: "ok", authentication: "prototype-email" });
});

test("Python child processes do not receive admin credentials", () => {
  assert.deepEqual(pythonEnvironment({
    BEARVISION_ADMIN_USERNAME: "admin",
    BEARVISION_ADMIN_PASSWORD: "secret",
    BEARVISION_SERVER_CONFIG: "config/server.yaml",
    PATH: "runtime-path",
  }), {
    BEARVISION_SERVER_CONFIG: "config/server.yaml",
    PATH: "runtime-path",
  });
});

test("the Server Control process fails before startup without admin credentials", async () => {
  const child = spawn(process.execPath, [fileURLToPath(new URL("./server.mjs", import.meta.url))], {
    cwd: process.cwd(),
    env: pythonEnvironment(process.env),
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => { stdout += chunk; });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  const exitCode = await new Promise((resolvePromise, reject) => {
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error("Server Control did not fail closed"));
    }, 3_000);
    child.once("error", reject);
    child.once("close", (code) => {
      clearTimeout(timer);
      resolvePromise(code);
    });
  });

  assert.notEqual(exitCode, 0);
  assert.equal(stdout.includes("BearVision Server Control:"), false);
  assert.match(stderr, /invalid Server Control authentication configuration/);
});
