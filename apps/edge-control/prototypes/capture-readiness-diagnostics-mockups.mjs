// THROWAWAY helper for capturing the readiness/diagnostics split prototype.
import { mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { chromium } from "@playwright/test";

const outputDir = path.resolve("../../artifacts/edge-control-redesign/screenshots");
await mkdir(outputDir, { recursive: true });

const channel = process.platform === "win32" && !existsSync(chromium.executablePath())
  ? "chrome"
  : undefined;
const browser = await chromium.launch({ headless: true, ...(channel ? { channel } : {}) });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });

const captures = [
  ["basis-command-overview.png", "variant=command&round=0&view=overview"],
  ["basis-workflow-overview.png", "variant=workflow&round=0&view=overview"],
  ["basis-focus-overview.png", "variant=focus&round=0&view=overview"],
  ["iteration-1-overview.png", "variant=command&round=1&view=overview"],
  ["iteration-2-overview.png", "variant=command&round=2&view=overview"],
  ["iteration-2-readiness.png", "variant=command&round=2&view=readiness"],
  ["iteration-2-logs.png", "variant=command&round=2&view=logs"],
];

for (const [filename, query] of captures) {
  await page.goto(`http://127.0.0.1:5173/prototypes/readiness-diagnostics-split-prototype.html?${query}`);
  if (filename.startsWith("iteration-2")) {
    await page.locator(".prototype-bar").evaluate((element) => { element.style.display = "none"; });
    await page.locator(".review-note").evaluateAll((elements) => elements.forEach((element) => { element.style.display = "none"; }));
  }
  await page.screenshot({ path: path.join(outputDir, filename), fullPage: true });
}

await browser.close();
