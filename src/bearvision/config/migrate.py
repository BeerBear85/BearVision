"""Migration helpers for pre-versioned BearVision configuration files."""

from __future__ import annotations

import argparse
import configparser
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .models import EdgeConfig


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_if_present(section: configparser.SectionProxy, key: str, target: dict, target_key: str, cast) -> None:
    if key in section:
        target[target_key] = cast(section[key])


def migrate_edge_data(ini_path: str | Path, yaml_path: str | Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Merge legacy INI and YAML edge settings; YAML takes precedence."""

    data = EdgeConfig(
        config_schema_version="2.0",
        config_kind="bearvision-edge",
    ).model_dump(mode="json")
    warnings: list[str] = []

    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    edge = parser["EDGE_APPLICATION"] if parser.has_section("EDGE_APPLICATION") else None
    if edge:
        _set_if_present(edge, "recording_duration", data["recording"], "post_detection_duration_s", float)
        _set_if_present(edge, "hindsight_mode_enabled", data["recording"], "hindsight_enabled", _as_bool)
        _set_if_present(edge, "yolo_enabled", data["detection"], "enabled", _as_bool)
        _set_if_present(edge, "yolo_model", data["detection"], "model", str)
        _set_if_present(edge, "detection_confidence_threshold", data["detection"], "confidence_threshold", float)
        _set_if_present(edge, "detection_cooldown", data["detection"], "cooldown_s", float)
        _set_if_present(edge, "stream_max_fps", data["performance"], "max_fps", int)
        _set_if_present(edge, "stream_buffer_drain", data["performance"], "buffer_drain", _as_bool)
        _set_if_present(edge, "stream_callback_queue_size", data["performance"], "callback_queue_size", int)
        _set_if_present(edge, "max_error_restarts", data["error_recovery"], "max_restarts", int)
        _set_if_present(edge, "error_restart_delay", data["error_recovery"], "restart_delay_s", float)
        _set_if_present(edge, "enable_ble_logging", data["features"], "ble_logging", _as_bool)
        _set_if_present(edge, "enable_cloud_upload", data["features"], "cloud_upload", _as_bool)

    if parser.has_section("BOX"):
        _set_if_present(parser["BOX"], "root_folder", data["storage"], "root_folder", str)
    if parser.has_section("STORAGE_COMMON"):
        shared = parser["STORAGE_COMMON"]
        _set_if_present(shared, "secret_key_name", data["storage"], "credential_env", str)
        _set_if_present(shared, "secret_key_name_2", data["storage"], "secondary_credential_env", str)

    ignored = set(parser.sections()) - {"EDGE_APPLICATION", "BOX", "STORAGE_COMMON", "WEB_STORIES"}
    warnings.extend(f"legacy INI section not used by edge config: {name}" for name in sorted(ignored))

    with Path(yaml_path).open(encoding="utf-8") as stream:
        legacy_yaml = yaml.safe_load(stream) or {}

    recording = legacy_yaml.get("recording", {})
    if "post_detection_duration" in recording:
        data["recording"]["post_detection_duration_s"] = recording["post_detection_duration"]
    if "hindsight_enabled" in recording:
        data["recording"]["hindsight_enabled"] = recording["hindsight_enabled"]

    detection = legacy_yaml.get("detection", {})
    yaml_detection_map = {
        "yolo_enabled": "enabled",
        "yolo_model": "model",
        "confidence_threshold": "confidence_threshold",
        "cooldown": "cooldown_s",
    }
    for source_key, target_key in yaml_detection_map.items():
        if source_key in detection:
            data["detection"][target_key] = detection[source_key]

    performance = legacy_yaml.get("performance", {})
    if "stream_max_fps" in performance:
        data["performance"]["max_fps"] = performance["stream_max_fps"]

    recovery = legacy_yaml.get("error_recovery", {})
    if "max_restarts" in recovery:
        data["error_recovery"]["max_restarts"] = recovery["max_restarts"]
    if "restart_delay" in recovery:
        data["error_recovery"]["restart_delay_s"] = recovery["restart_delay"]

    threads = legacy_yaml.get("threads", {})
    if "enable_ble_logging" in threads:
        data["features"]["ble_logging"] = threads["enable_ble_logging"]
    if "enable_cloud_upload" in threads:
        data["features"]["cloud_upload"] = threads["enable_cloud_upload"]
    if "log_level" in legacy_yaml.get("system", {}):
        data["system"]["log_level"] = legacy_yaml["system"]["log_level"]

    known_yaml = {"recording", "detection", "performance", "error_recovery", "threads", "system"}
    warnings.extend(f"legacy YAML section not used by edge config: {name}" for name in sorted(set(legacy_yaml) - known_yaml))

    validated = EdgeConfig.model_validate(deepcopy(data))
    return validated.model_dump(mode="json"), tuple(warnings)


def add_version_header(data: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """Add a version header to a previously unversioned YAML mapping."""

    if "config_schema_version" in data or "config_kind" in data:
        raise ValueError("configuration is already versioned")
    return {"config_schema_version": "2.0", "config_kind": kind, **data}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write YAML without overwriting an existing migration target."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"migration target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate BearVision configuration")
    parser.add_argument("source", type=Path, help="legacy INI or YAML file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["edge", "ble-test", "training", "annotation"])
    parser.add_argument("--edge-yaml", type=Path, help="legacy edge YAML merged after the INI")
    args = parser.parse_args()

    if args.kind == "edge":
        if not args.edge_yaml:
            parser.error("--edge-yaml is required for edge migration")
        data, warnings = migrate_edge_data(args.source, args.edge_yaml)
        for warning in warnings:
            print(f"WARNING: {warning}")
    else:
        with args.source.open(encoding="utf-8") as stream:
            legacy = yaml.safe_load(stream) or {}
        data = add_version_header(legacy, kind=f"bearvision-{args.kind}")

    write_yaml(args.output, data)
    print(f"Migrated configuration written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
