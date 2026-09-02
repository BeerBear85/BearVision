"""Operator-facing hardware readiness checks for Edge Control."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Coroutine, Literal

from bearvision.adapters import (
    BleakKBeaconSource,
    FfmpegVideoClipper,
    GoProCameraAdapter,
    OpenCvPreviewFrameSource,
    SystemClock,
)
from bearvision.config import EdgeConfig, load_edge_config

from .hardware_readiness import PhysicalHandshake, PhysicalReadinessHandshake


ReadinessStatus = Literal["pass", "warning", "fail"]


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    status: ReadinessStatus
    evidence: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warning", "fail"}:
            raise ValueError("readiness status must be pass, warning or fail")
        if not self.evidence:
            raise ValueError("readiness evidence must not be empty")


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    check_id: str
    label: str
    status: ReadinessStatus
    critical: bool
    evidence: str
    corrective_action: str


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    readiness_schema_version: str
    checked_at: str
    blocking: bool
    warning_ids: tuple[str, ...]
    checks: tuple[ReadinessCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _CheckDefinition:
    check_id: str
    label: str
    critical: bool
    corrective_action: str
    probe: Callable[[], ProbeOutcome]


def _pass(evidence: str) -> ProbeOutcome:
    return ProbeOutcome("pass", evidence)


def _fail_from(operation: Callable[[], str]) -> ProbeOutcome:
    try:
        return _pass(operation())
    except Exception as exc:
        return ProbeOutcome("fail", str(exc))


def _fail_from_async(operation: Callable[[], Coroutine[Any, Any, str]]) -> ProbeOutcome:
    try:
        result: str = asyncio.run(operation())
        return _pass(str(result))
    except Exception as exc:
        return ProbeOutcome("fail", str(exc))


def _runtime_probe(config_path: Path) -> ProbeOutcome:
    def inspect() -> str:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError(
                f"Python 3.12 is required; running {sys.version_info.major}.{sys.version_info.minor}"
            )
        load_edge_config(config_path)
        return f"Python {sys.version_info.major}.{sys.version_info.minor}; configuration valid"

    return _fail_from(inspect)


def _model_probe(config: EdgeConfig) -> ProbeOutcome:
    def inspect() -> str:
        from bearvision.integrations.opencv_dnn import DnnHandler

        detector = DnnHandler(config.detection.model)
        detector.init()
        return f"YOLO model loaded: {detector.model}"

    return _fail_from(inspect)


def _media_tools_probe(config: EdgeConfig) -> ProbeOutcome:
    def inspect() -> str:
        tools = FfmpegVideoClipper(config.clip_extraction)
        missing = [
            executable
            for executable in (tools.ffmpeg_path, tools.ffprobe_path)
            if not Path(executable).is_file() and shutil.which(executable) is None
        ]
        if missing:
            raise RuntimeError(f"Media executable is unavailable: {', '.join(missing)}")
        return f"FFmpeg: {tools.ffmpeg_path}; FFprobe: {tools.ffprobe_path}"

    return _fail_from(inspect)


def _storage_probe(path: Path) -> ProbeOutcome:
    def inspect() -> str:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".bearvision-readiness-", dir=path, delete=True) as item:
            item.write(b"ready")
            item.flush()
        free_bytes = shutil.disk_usage(path).free
        if free_bytes < 1024**3:
            raise RuntimeError(f"Only {free_bytes // (1024**2)} MiB free at {path}")
        return f"Writable; {free_bytes // (1024**3)} GiB free at {path}"

    return _fail_from(inspect)


def _physical_handshake(
    config: EdgeConfig,
    *,
    capture_dir: str | Path,
) -> PhysicalReadinessHandshake:
    clock = SystemClock()

    def camera_factory() -> GoProCameraAdapter:
        from bearvision.integrations.async_gopro import AsyncGoProController

        return GoProCameraAdapter(
            AsyncGoProController(),
            clock,
            capture_dir,
            hindsight_enabled=config.recording.hindsight_enabled,
            hindsight_duration_s=config.recording.hindsight_duration_s,
        )

    return PhysicalReadinessHandshake(
        camera_factory=camera_factory,
        frame_source_factory=lambda: OpenCvPreviewFrameSource(
            clock,
            max_fps=1,
            queue_size=1,
            drain_old_frames=True,
        ),
        ble_source_factory=BleakKBeaconSource,
        camera_timeout_s=config.readiness.camera_preview_timeout_s,
        ble_scan_duration_s=config.readiness.ble_scan_duration_s,
        cleanup_timeout_s=config.readiness.cleanup_timeout_s,
    )


def _cloud_storage_probe(config: EdgeConfig) -> ProbeOutcome:
    if not config.features.cloud_upload:
        return _pass("Cloud upload is disabled in the active configuration")

    def inspect() -> str:
        from bearvision.integrations.box_handler import BoxHandler

        box_config = {
            "STORAGE_COMMON": {
                "secret_key_name": config.storage.credential_env,
                "secret_key_name_2": config.storage.secondary_credential_env or "",
            },
            "BOX": {"root_folder": config.storage.root_folder},
        }
        handler = BoxHandler(box_config)
        handler.connect()
        return f"Box root is reachable: {config.storage.root_folder}"

    return _fail_from(inspect)


def check_edge_readiness(
    config_path: str | Path,
    *,
    capture_dir: str | Path,
    scratch_dir: str | Path,
    probe_overrides: dict[str, Callable[[], ProbeOutcome]] | None = None,
    hardware_handshake: PhysicalHandshake | None = None,
) -> ReadinessReport:
    """Run every readiness check and return one operator-facing report."""

    config_path = Path(config_path)
    try:
        config = load_edge_config(config_path)
    except Exception:
        config = None
    if hardware_handshake is None and config is not None:
        hardware_handshake = _physical_handshake(config, capture_dir=capture_dir)
    def unavailable_without_config() -> ProbeOutcome:
        return ProbeOutcome(
            "fail", "The Edge configuration is invalid, so this check could not run"
        )
    definitions = (
        _CheckDefinition(
            "runtime", "Runtime and configuration", True,
            "Install Python 3.12 and correct the active Edge configuration.",
            lambda: _runtime_probe(config_path),
        ),
        _CheckDefinition(
            "model", "Detection model", True,
            "Restore the configured ONNX model or select an installed model.",
            (lambda: _model_probe(config)) if config else unavailable_without_config,
        ),
        _CheckDefinition(
            "media_tools", "Media tools", True,
            "Install or configure working FFmpeg and FFprobe executables.",
            (lambda: _media_tools_probe(config)) if config else unavailable_without_config,
        ),
        _CheckDefinition(
            "capture_storage", "Capture storage", True,
            "Make the capture directory writable and free at least 1 GiB.",
            lambda: _storage_probe(Path(capture_dir)),
        ),
        _CheckDefinition(
            "scratch_storage", "Working storage", True,
            "Make the scratch directory writable and free at least 1 GiB.",
            lambda: _storage_probe(Path(scratch_dir)),
        ),
        _CheckDefinition(
            "camera", "GoPro camera", True,
            (
                "Connect and power on the GoPro over USB, close other preview consumers, "
                "verify USB networking, then run readiness again."
            ),
            (
                lambda: _fail_from_async(hardware_handshake.check_camera_preview)
                if hardware_handshake is not None
                else unavailable_without_config()
            ),
        ),
        _CheckDefinition(
            "ble", "Bluetooth and BearTag scanning", True,
            (
                "Enable the Bluetooth adapter, allow scanning permission, close other BLE "
                "tools, then run readiness again."
            ),
            (
                lambda: _fail_from_async(hardware_handshake.check_ble_scanner)
                if hardware_handshake is not None
                else unavailable_without_config()
            ),
        ),
        _CheckDefinition(
            "cloud_storage", "Upload storage", True,
            "Check Box credentials and network connectivity, then run readiness again.",
            (lambda: _cloud_storage_probe(config)) if config else unavailable_without_config,
        ),
    )
    overrides = probe_overrides or {}
    checks = tuple(
        ReadinessCheck(
            check_id=item.check_id,
            label=item.label,
            status=(outcome := overrides.get(item.check_id, item.probe)()).status,
            critical=item.critical,
            evidence=outcome.evidence,
            corrective_action=item.corrective_action,
        )
        for item in definitions
    )
    return ReadinessReport(
        readiness_schema_version="1.0",
        checked_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        blocking=any(check.critical and check.status == "fail" for check in checks),
        warning_ids=tuple(check.check_id for check in checks if check.status == "warning"),
        checks=checks,
    )
