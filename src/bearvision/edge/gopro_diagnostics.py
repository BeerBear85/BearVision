"""Non-destructive, layered diagnostics for a wired GoPro connection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import socket
import subprocess
from typing import Any, Callable, Literal
from urllib.request import urlopen


DiagnosticStatus = Literal["pass", "fail", "unavailable"]
GOPRO_USB_VENDOR_ID = "2672"
DEFAULT_GOPRO_TARGET = "172.24.106.51"
DEFAULT_GOPRO_PORT = 8080


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    check_id: str
    label: str
    status: DiagnosticStatus
    evidence: str
    corrective_action: str | None = None


@dataclass(frozen=True, slots=True)
class GoProDiagnosticReport:
    diagnostics_schema_version: str
    checked_at: str
    target: str
    status: Literal["pass", "fail"]
    summary: str
    checks: tuple[DiagnosticCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _usb_check(sysfs_root: Path) -> DiagnosticCheck:
    if not sysfs_root.is_dir():
        return DiagnosticCheck(
            "usb_device",
            "USB device detection",
            "unavailable",
            f"USB device details are unavailable at {sysfs_root}.",
            "Run this diagnostic on the Linux Edge computer.",
        )

    try:
        usb_entries = tuple(sysfs_root.iterdir())
    except OSError as exc:
        return DiagnosticCheck(
            "usb_device",
            "USB device detection",
            "unavailable",
            f"The USB bus could not be inspected: {exc}",
            "Check the Edge service permissions for /sys/bus/usb/devices.",
        )

    devices: list[str] = []
    for device in usb_entries:
        if not device.is_dir():
            continue
        vendor = _read_optional(device / "idVendor").lower()
        manufacturer = _read_optional(device / "manufacturer")
        product = _read_optional(device / "product")
        if vendor == GOPRO_USB_VENDOR_ID or "gopro" in f"{manufacturer} {product}".lower():
            description = " ".join(part for part in (manufacturer, product) if part) or "GoPro"
            devices.append(f"{description} (USB vendor {vendor or 'unknown'})")

    if devices:
        return DiagnosticCheck(
            "usb_device",
            "USB device detection",
            "pass",
            f"Detected {', '.join(devices)}.",
        )
    return DiagnosticCheck(
        "usb_device",
        "USB device detection",
        "fail",
        "The Edge computer can read the USB bus, but no GoPro device was detected.",
        "Power on the GoPro, reconnect the USB cable, and try another data-capable cable or port.",
    )


def _default_command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, check=False, text=True, timeout=3)


def _network_check(
    target: str,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> DiagnosticCheck:
    try:
        result = command_runner(["ip", "-json", "address", "show"])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"ip exited with {result.returncode}")
        links = json.loads(result.stdout)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        return DiagnosticCheck(
            "usb_network",
            "USB network interface",
            "unavailable",
            f"Network interface details could not be read: {exc}",
            "Verify that the iproute2 tools are installed on the Edge computer.",
        )

    target_address = ipaddress.ip_address(target)
    matching: list[str] = []
    available: list[str] = []
    for link in links:
        name = str(link.get("ifname", "unknown"))
        for address in link.get("addr_info", []):
            local = address.get("local")
            prefix = address.get("prefixlen")
            if address.get("family") != "inet" or local is None or prefix is None:
                continue
            available.append(f"{name}={local}/{prefix}")
            try:
                if target_address in ipaddress.ip_interface(f"{local}/{prefix}").network:
                    matching.append(f"{name} ({local}/{prefix})")
            except ValueError:
                continue

    if matching:
        return DiagnosticCheck(
            "usb_network",
            "USB network interface",
            "pass",
            f"Camera-compatible network interface found: {', '.join(matching)}.",
        )
    evidence = "No IPv4 interface can reach the camera subnet."
    if available:
        evidence += f" Available interfaces: {', '.join(available)}."
    return DiagnosticCheck(
        "usb_network",
        "USB network interface",
        "fail",
        evidence,
        "Unlock the GoPro, accept USB networking if prompted, then reconnect the cable.",
    )


def _tcp_check(
    target: str,
    port: int,
    tcp_connect: Callable[..., Any],
) -> DiagnosticCheck:
    try:
        connection = tcp_connect((target, port), timeout=2)
        connection.close()
    except OSError as exc:
        return DiagnosticCheck(
            "camera_tcp",
            "Camera API connection",
            "fail",
            f"TCP connection to {target}:{port} failed: {exc}",
            "Check the USB network interface and confirm that the GoPro is powered on and unlocked.",
        )
    return DiagnosticCheck(
        "camera_tcp",
        "Camera API connection",
        "pass",
        f"TCP connection to {target}:{port} succeeded.",
    )


def _http_check(
    target: str,
    port: int,
    open_url: Callable[..., Any],
) -> DiagnosticCheck:
    endpoint = f"http://{target}:{port}/gopro/camera/state"
    try:
        with open_url(endpoint, timeout=3) as response:
            status = int(getattr(response, "status", 200))
            body = response.read(64 * 1024)
        if not 200 <= status < 300:
            raise RuntimeError(f"HTTP {status}")
        json.loads(body)
    except (OSError, ValueError, RuntimeError) as exc:
        return DiagnosticCheck(
            "camera_http",
            "GoPro HTTP communication",
            "fail",
            f"The camera state endpoint did not return valid JSON: {exc}",
            "Restart the GoPro after the USB and network checks pass, then run diagnostics again.",
        )
    return DiagnosticCheck(
        "camera_http",
        "GoPro HTTP communication",
        "pass",
        f"The GoPro state endpoint responded with HTTP {status} and valid JSON.",
    )


def diagnose_gopro(
    *,
    target: str = DEFAULT_GOPRO_TARGET,
    port: int = DEFAULT_GOPRO_PORT,
    sysfs_root: Path = Path("/sys/bus/usb/devices"),
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _default_command_runner,
    tcp_connect: Callable[..., Any] = socket.create_connection,
    open_url: Callable[..., Any] = urlopen,
) -> GoProDiagnosticReport:
    """Inspect each connection layer without changing camera state."""

    checks = (
        _usb_check(sysfs_root),
        _network_check(target, command_runner),
        _tcp_check(target, port, tcp_connect),
        _http_check(target, port, open_url),
    )
    by_id = {check.check_id: check for check in checks}
    if by_id["camera_http"].status == "pass":
        summary = "The Edge computer can communicate with the GoPro HTTP API."
        status: Literal["pass", "fail"] = "pass"
    elif by_id["usb_device"].status == "fail":
        summary = "The Edge computer cannot currently detect a GoPro on the USB bus."
        status = "fail"
    elif by_id["usb_network"].status == "fail":
        summary = "The GoPro is visible over USB, but its USB network connection is not ready."
        status = "fail"
    elif by_id["camera_tcp"].status == "fail":
        summary = "The camera network exists, but the GoPro API port does not respond."
        status = "fail"
    else:
        summary = "GoPro communication failed; some local diagnostic details were unavailable."
        status = "fail"

    return GoProDiagnosticReport(
        diagnostics_schema_version="1.0",
        checked_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        target=f"{target}:{port}",
        status=status,
        summary=summary,
        checks=checks,
    )
