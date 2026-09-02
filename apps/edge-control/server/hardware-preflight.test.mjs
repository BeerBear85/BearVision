import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { mkdtempSync } from "node:fs";

import { assertGoProUsbConnected, findGoProUsbDevice } from "./hardware-preflight.mjs";

function usbDevice(root, name, fields) {
  const device = join(root, name);
  mkdirSync(device);
  for (const [field, value] of Object.entries(fields)) {
    writeFileSync(join(device, field), `${value}\n`, "utf8");
  }
}

test("hardware preflight finds a connected GoPro USB device", () => {
  const root = mkdtempSync(join(tmpdir(), "bearvision-usb-"));
  usbDevice(root, "1-1.1", {
    idVendor: "2672",
    manufacturer: "GoPro",
    product: "HERO12 Black",
  });

  assert.deepEqual(findGoProUsbDevice(root), {
    sysfsName: "1-1.1",
    product: "HERO12 Black",
  });
  assert.doesNotThrow(() => assertGoProUsbConnected(root));
});

test("hardware preflight fails immediately with an actionable USB error", () => {
  const root = mkdtempSync(join(tmpdir(), "bearvision-usb-"));
  usbDevice(root, "1-1", { idVendor: "2109", manufacturer: "VIA Labs" });

  assert.throws(
    () => assertGoProUsbConnected(root),
    /GoPro USB device is not connected or powered on/,
  );
});
