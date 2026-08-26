"""Versioned behavioural scenario contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from .models import Vector3


class TimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    at_s: float = Field(ge=0)
    event: Literal["tag_enters_range", "tag_observation", "person_detected"]
    payload: dict[str, Any] = Field(default_factory=dict)


class ScenarioFaults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    camera_capture: bool = False
    storage_upload: bool = False


class ScenarioExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rider_id: str | None = None
    rider_ids: tuple[str, ...] | None = None
    bear_tag_id: str | None = None
    bear_tag_ids: tuple[str, ...] | None = None
    assignment_status: Literal["assigned", "unassigned", "ambiguous"] | None = None
    capture_triggered: bool | None = None
    clip_uploaded: bool | None = None
    minimum_person_detections: int | None = Field(default=None, ge=0)
    first_detection_between_s: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_detection_window(self) -> "ScenarioExpectation":
        if self.rider_id is not None and self.rider_ids:
            raise ValueError("expectation must declare rider_id or rider_ids, not both")
        if self.rider_ids and len(set(self.rider_ids)) != len(self.rider_ids):
            raise ValueError("expected rider_ids must be unique")
        if self.bear_tag_id is not None and self.bear_tag_ids:
            raise ValueError("expectation must declare bear_tag_id or bear_tag_ids, not both")
        if self.bear_tag_ids and len(set(self.bear_tag_ids)) != len(self.bear_tag_ids):
            raise ValueError("expected bear_tag_ids must be unique")
        if self.rider_ids and self.bear_tag_ids and len(self.rider_ids) != len(self.bear_tag_ids):
            raise ValueError("expected rider_ids and bear_tag_ids must have equal lengths")
        if self.first_detection_between_s is not None:
            start_s, end_s = self.first_detection_between_s
            if start_s < 0 or end_s < start_s:
                raise ValueError("first_detection_between_s must be an ordered positive window")
        return self


class ScenarioComponents(BaseModel):
    """Select the implementation behind each BearVision port for one scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    frames: Literal["synthetic", "video", "gopro"] = "synthetic"
    detector: Literal["declared", "yolo"] = "declared"
    bear_tag: Literal["synthetic", "ble"] = "synthetic"
    camera: Literal["simulated", "simulated_gopro", "gopro"] = "simulated"
    storage: Literal["memory", "box"] = "memory"

    @model_validator(mode="before")
    @classmethod
    def migrate_recorded_video_camera(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("camera") == "recorded_video":
            return {**data, "camera": "simulated_gopro"}
        return data


class ScenarioVideo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    sample_fps: float = Field(default=5.0, gt=0, le=60)


class ScenarioDetector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(default="yolov8n", min_length=1)
    confidence_threshold: float = Field(default=0.6, ge=0, le=1)


class SyntheticMotionWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    acceleration_mps2: Vector3
    rssi_dbm: int | None = Field(default=None, ge=-127, le=20)

    @model_validator(mode="after")
    def validate_window(self) -> "SyntheticMotionWindow":
        if self.end_s <= self.start_s:
            raise ValueError("motion window end_s must exceed start_s")
        return self


class SyntheticBearTagSample(BaseModel):
    """One explicit synthetic BearTag advertisement and its simulation truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    at_s: float = Field(ge=0)
    rssi_dbm: int = Field(ge=-127, le=20)
    acceleration_mps2: Vector3
    battery_voltage_mv: int | None = Field(default=None, ge=0, le=10_000)
    source_frame: int | None = Field(default=None, ge=0)
    source_distance_m: float | None = Field(default=None, gt=0)


class SyntheticBearTagSeries(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tag_id: str = Field(min_length=1)
    rider_id: str = Field(min_length=1)
    start_s: float = Field(default=0, ge=0)
    end_s: float = Field(gt=0)
    sample_rate_hz: float = Field(default=10.0, gt=0, le=100)
    rssi_dbm: int = Field(default=-60, ge=-127, le=20)
    baseline_acceleration_mps2: Vector3 = Field(
        default_factory=lambda: Vector3(x=0, y=0, z=9.80665)
    )
    motion_windows: tuple[SyntheticMotionWindow, ...] = ()
    samples: tuple[SyntheticBearTagSample, ...] = ()

    @model_validator(mode="after")
    def validate_series(self) -> "SyntheticBearTagSeries":
        if self.end_s <= self.start_s:
            raise ValueError("synthetic BearTag end_s must exceed start_s")
        for window in self.motion_windows:
            if window.start_s < self.start_s or window.end_s > self.end_s:
                raise ValueError("motion windows must stay inside the BearTag series")
        if self.samples and self.motion_windows:
            raise ValueError("explicit BearTag samples cannot be combined with motion windows")
        previous_at_s: float | None = None
        for sample in self.samples:
            if sample.at_s < self.start_s or sample.at_s > self.end_s:
                raise ValueError("explicit samples must stay inside the BearTag series")
            if previous_at_s is not None and sample.at_s <= previous_at_s:
                raise ValueError("explicit BearTag samples must be strictly time ordered")
            previous_at_s = sample.at_s
        return self


class GeneratedScenarioSource(BaseModel):
    """Inputs and physical assumptions used to generate a scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    generator: Literal["blender-motion-v1"]
    motion_path: str | None = Field(default=None, min_length=1)
    motion_paths: tuple[str, ...] | None = None
    camera_path: str | None = Field(default=None, min_length=1)
    reference_rssi_dbm_at_1m: int = Field(ge=-127, le=20)
    path_loss_exponent: float = Field(gt=0)
    gravity_mps2: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_motion_provenance(self) -> "GeneratedScenarioSource":
        if bool(self.motion_path) == bool(self.motion_paths):
            raise ValueError("generated source must declare motion_path or motion_paths")
        if self.motion_paths and any(not path for path in self.motion_paths):
            raise ValueError("generated motion paths must not be empty")
        return self


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_schema_version: Literal["2.0", "3.0", "3.1"]
    name: str = Field(min_length=1, max_length=150)
    seed: int = 0
    duration_s: float = Field(gt=0)
    timeline: tuple[TimelineEvent, ...]
    faults: ScenarioFaults = Field(default_factory=ScenarioFaults)
    expect: ScenarioExpectation = Field(default_factory=ScenarioExpectation)
    components: ScenarioComponents = Field(default_factory=ScenarioComponents)
    video: ScenarioVideo | None = None
    detector: ScenarioDetector = Field(default_factory=ScenarioDetector)
    synthetic_bear_tags: tuple[SyntheticBearTagSeries, ...] = ()
    generated_from: GeneratedScenarioSource | None = None

    @model_validator(mode="after")
    def validate_component_composition(self) -> "ScenarioDefinition":
        if self.scenario_schema_version == "2.0":
            if (
                self.video is not None
                or self.synthetic_bear_tags
                or self.generated_from is not None
            ):
                raise ValueError("video and generated BearTag series require scenario schema 3.x")
            return self
        if self.scenario_schema_version == "3.0" and (
            self.generated_from is not None
            or any(series.samples for series in self.synthetic_bear_tags)
        ):
            raise ValueError("generated provenance and explicit BearTag samples require schema 3.1")
        if self.components.frames == "video" and self.video is None:
            raise ValueError("video frame source requires a video configuration")
        if self.components.frames != "video" and self.video is not None:
            raise ValueError("video configuration requires components.frames=video")
        if self.components.detector == "yolo" and self.components.frames == "synthetic":
            raise ValueError("YOLO requires video or GoPro frames")
        if self.components.detector == "yolo" and any(
            item.event == "person_detected" for item in self.timeline
        ):
            raise ValueError("YOLO scenarios must not declare person_detected timeline events")
        if self.components.bear_tag == "ble" and self.synthetic_bear_tags:
            raise ValueError("physical BLE scenarios cannot include generated BearTag series")
        return self


def load_scenario(path: str | Path) -> ScenarioDefinition:
    """Load and strictly validate a YAML scenario."""

    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return ScenarioDefinition.model_validate(data)
