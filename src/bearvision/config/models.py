"""Version 1.0 configuration schema for the BearVision 3 edge system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecordingConfig(StrictConfigModel):
    post_detection_duration_s: float = Field(default=5.0, ge=0, le=300)
    hindsight_enabled: bool = True


class DetectionConfig(StrictConfigModel):
    enabled: bool = True
    model: str = Field(default="yolov8n", min_length=1)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    cooldown_s: float = Field(default=2.0, ge=0, le=300)


class PerformanceConfig(StrictConfigModel):
    max_fps: int = Field(default=30, ge=1, le=120)
    max_lag_ms: int = Field(default=250, ge=0, le=60_000)
    buffer_drain: bool = True
    callback_queue_size: int = Field(default=5, ge=1, le=100)


class ErrorRecoveryConfig(StrictConfigModel):
    max_restarts: int = Field(default=5, ge=0, le=100)
    restart_delay_s: float = Field(default=2.0, ge=0, le=600)


class FeatureConfig(StrictConfigModel):
    ble_logging: bool = True
    post_processing: bool = True
    cloud_upload: bool = True
    preview_stream: bool = True


class StorageConfig(StrictConfigModel):
    provider: Literal["box"] = "box"
    root_folder: str = Field(default="bearvision_files", min_length=1)
    credential_env: str = Field(default="STORAGE_CREDENTIALS_B64", min_length=1)
    secondary_credential_env: str | None = "STORAGE_CREDENTIALS_B64_2"


class SystemConfig(StrictConfigModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class TagRegistration(StrictConfigModel):
    tag_id: str = Field(min_length=1)
    rider_id: str = Field(min_length=1)
    enabled: bool = True


class EdgeConfig(StrictConfigModel):
    """Complete edge configuration; versioned separately from the application."""

    config_schema_version: Literal["1.0"]
    config_kind: Literal["bearvision-edge"]
    recording: RecordingConfig = RecordingConfig()
    detection: DetectionConfig = DetectionConfig()
    performance: PerformanceConfig = PerformanceConfig()
    error_recovery: ErrorRecoveryConfig = ErrorRecoveryConfig()
    features: FeatureConfig = FeatureConfig()
    storage: StorageConfig = StorageConfig()
    system: SystemConfig = SystemConfig()
    tag_registry: tuple[TagRegistration, ...] = ()


def load_edge_config(path: str | Path) -> EdgeConfig:
    """Load a strict versioned edge YAML file."""

    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return EdgeConfig.model_validate(data)
