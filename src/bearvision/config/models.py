"""Version 2.0 configuration schema for the BearVision 3 edge system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecordingConfig(StrictConfigModel):
    post_detection_duration_s: float = Field(default=5.0, ge=0, le=300)
    hindsight_enabled: bool = True
    hindsight_duration_s: Literal[15, 30] = 15


class ClipExtractionConfig(StrictConfigModel):
    engine: Literal["ffmpeg"] = "ffmpeg"
    video_codec: Literal["libx264"] = "libx264"
    audio_codec: Literal["aac"] = "aac"
    preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"] = (
        "veryfast"
    )
    crf: int = Field(default=20, ge=0, le=51)


class VirtualCameramanConfig(StrictConfigModel):
    """Tracking, crop and output-quality policy for processed Edge clips."""

    sample_fps: float = Field(default=10.0, gt=0, le=120)
    crop_width_ratio: float = Field(default=0.5, gt=0, le=1)
    output_width_px: int = Field(default=960, gt=0)
    output_height_px: int = Field(default=540, gt=0)
    process_noise_acceleration_px_s2: float = Field(default=800.0, gt=0)
    velocity_damping_time_constant_s: float = Field(default=5.0, gt=0)
    minimum_measurement_std_px: float = Field(default=2.0, gt=0)
    innovation_gate_chi2: float = Field(default=9.210340371976184, gt=0)
    maximum_bootstrap_speed_px_s: float = Field(default=3_000.0, gt=0)
    camera_cutoff_hz: float = Field(default=1.25, gt=0)
    length_adjustment_padding_s: float = Field(default=1.0, ge=0)
    output_crf: int = Field(default=18, ge=0, le=51)

    @model_validator(mode="after")
    def validate_h264_dimensions(self) -> "VirtualCameramanConfig":
        if self.output_width_px % 2 or self.output_height_px % 2:
            raise ValueError("H.264 output dimensions must be even")
        return self


class DetectionConfig(StrictConfigModel):
    enabled: bool = True
    model: str = Field(default="yolov8n", min_length=1)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    cooldown_s: float = Field(default=2.0, ge=0, le=300)


class AssignmentConfig(StrictConfigModel):
    """Initial BearTag fusion policy; values require field-data calibration."""

    minimum_observation_count: int = Field(default=2, ge=1, le=1000)
    minimum_motion_delta_mps2: float = Field(default=2.0, ge=0, le=100)
    motion_full_scale_mps2: float = Field(default=12.0, gt=0, le=100)
    minimum_rssi_dbm: int = Field(default=-85, ge=-127, le=20)
    rssi_full_scale_dbm: int = Field(default=-40, ge=-127, le=20)
    motion_weight: float = Field(default=0.7, ge=0, le=1)
    rssi_weight: float = Field(default=0.3, ge=0, le=1)
    minimum_score_margin: float = Field(default=0.12, ge=0, le=1)

    @model_validator(mode="after")
    def validate_scales_and_weights(self) -> "AssignmentConfig":
        if self.motion_full_scale_mps2 <= self.minimum_motion_delta_mps2:
            raise ValueError("motion_full_scale_mps2 must exceed minimum_motion_delta_mps2")
        if self.rssi_full_scale_dbm <= self.minimum_rssi_dbm:
            raise ValueError("rssi_full_scale_dbm must exceed minimum_rssi_dbm")
        if abs(self.motion_weight + self.rssi_weight - 1.0) > 1e-9:
            raise ValueError("motion_weight and rssi_weight must sum to 1.0")
        return self


class PerformanceConfig(StrictConfigModel):
    max_fps: int = Field(default=30, ge=1, le=120)
    buffer_drain: bool = True
    callback_queue_size: int = Field(default=5, ge=1, le=100)


class ErrorRecoveryConfig(StrictConfigModel):
    max_restarts: int = Field(default=5, ge=0, le=100)
    restart_delay_s: float = Field(default=2.0, ge=0, le=600)


class HardwareReadinessConfig(StrictConfigModel):
    """Time bounds for non-destructive physical preflight handshakes."""

    camera_preview_timeout_s: float = Field(default=12.0, gt=0, le=60)
    ble_scan_duration_s: float = Field(default=2.0, gt=0, le=30)
    cleanup_timeout_s: float = Field(default=3.0, gt=0, le=30)


class FeatureConfig(StrictConfigModel):
    ble_logging: bool = True
    cloud_upload: bool = True


class StorageConfig(StrictConfigModel):
    provider: Literal["box"] = "box"
    root_folder: str = Field(default="bearvision_files", min_length=1)
    credential_env: str = Field(default="STORAGE_CREDENTIALS_B64", min_length=1)
    secondary_credential_env: str | None = "STORAGE_CREDENTIALS_B64_2"


class SystemConfig(StrictConfigModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    device_id: str = Field(default="edge-1", min_length=1, max_length=128)


class TagRegistration(StrictConfigModel):
    tag_id: str = Field(min_length=1)
    rider_id: str = Field(min_length=1)
    enabled: bool = True


class EdgeConfig(StrictConfigModel):
    """Complete edge configuration; versioned separately from the application."""

    config_schema_version: Literal["3.0"]
    config_kind: Literal["bearvision-edge"]
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    clip_extraction: ClipExtractionConfig = Field(default_factory=ClipExtractionConfig)
    virtual_cameraman: VirtualCameramanConfig = Field(
        default_factory=VirtualCameramanConfig
    )
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    error_recovery: ErrorRecoveryConfig = Field(default_factory=ErrorRecoveryConfig)
    readiness: HardwareReadinessConfig = Field(default_factory=HardwareReadinessConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)


class WorkerConfig(StrictConfigModel):
    poll_interval_s: float = Field(default=5.0, ge=0.1, le=3600)
    retry_delay_s: float = Field(default=2.0, ge=0, le=600)


class AdminConfig(StrictConfigModel):
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=4320, ge=1024, le=65535)


class ServerConfig(StrictConfigModel):
    config_schema_version: Literal["1.0"]
    config_kind: Literal["bearvision-server"]
    assignment: AssignmentConfig = Field(default_factory=AssignmentConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    registry_path: Path = Path("data/server/user-registry.json")
    scratch_dir: Path = Path("temp/server-box")
    local_queue_root: Path | None = None


def load_edge_config(path: str | Path) -> EdgeConfig:
    """Load a strict versioned edge YAML file."""

    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return EdgeConfig.model_validate(data)


def load_server_config(path: str | Path) -> ServerConfig:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return ServerConfig.model_validate(data)
