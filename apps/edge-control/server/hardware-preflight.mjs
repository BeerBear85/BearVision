import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const GOPRO_USB_VENDOR_ID = "2672";

function readField(deviceRoot, field) {
  const path = join(deviceRoot, field);
  return existsSync(path) ? readFileSync(path, "utf8").trim() : "";
}

export function findGoProUsbDevice(sysfsRoot = "/sys/bus/usb/devices") {
  if (!existsSync(sysfsRoot) || !statSync(sysfsRoot).isDirectory()) return null;
  for (const sysfsName of readdirSync(sysfsRoot)) {
    const deviceRoot = join(sysfsRoot, sysfsName);
    if (!statSync(deviceRoot).isDirectory()) continue;
    const vendor = readField(deviceRoot, "idVendor").toLowerCase();
    const manufacturer = readField(deviceRoot, "manufacturer").toLowerCase();
    if (vendor !== GOPRO_USB_VENDOR_ID && manufacturer !== "gopro") continue;
    return {
      sysfsName,
      product: readField(deviceRoot, "product") || "GoPro camera",
    };
  }
  return null;
}

export function assertGoProUsbConnected(sysfsRoot = "/sys/bus/usb/devices") {
  if (!existsSync(sysfsRoot)) return null;
  const camera = findGoProUsbDevice(sysfsRoot);
  if (!camera) {
    throw new Error(
      "GoPro USB device is not connected or powered on. Check camera power and the USB data cable, then retry.",
    );
  }
  return camera;
}
