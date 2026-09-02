import { existsSync } from "node:fs";

import { chromium, defineConfig } from "@playwright/test";

const browserChannel = process.env.BEARVISION_PLAYWRIGHT_CHANNEL
  ?? (process.platform === "win32" && !existsSync(chromium.executablePath()) ? "chrome" : undefined);

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  outputDir: "../../temp/playwright-edge-control",
  use: {
    browserName: "chromium",
    ...(browserChannel ? { channel: browserChannel } : {}),
    headless: true,
    trace: "retain-on-failure",
  },
});
