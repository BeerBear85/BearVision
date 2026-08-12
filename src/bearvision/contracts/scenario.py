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
    assignment_status: Literal["assigned", "unassigned", "ambiguous"] | None = None
    capture_triggered: bool | None = None
    clip_uploaded: bool | None = None
    minimum_person_detections: int | None = Field(default=None, ge=0)
    first_detection_between_s: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_detection_window(self) -> "ScenarioExpectation":
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
    camera: Literal["simulated", "recorded_video", "gopro"] = "simulated"
    storage: Literal["memory", "box"] = "memory"


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

    @model_validator(mode="after")
    def validate_series(self) -> "SyntheticBearTagSeries":
        if self.end_s <= self.start_s:
            raise ValueError("synthetic BearTag end_s must exceed start_s")
        for window in self.motion_windows:
            if window.start_s < self.start_s or window.end_s > self.end_s:
                raise ValueError("motion windows must stay inside the BearTag series")
        return self


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_schema_version: Literal["2.0", "3.0"]
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

    @model_validator(mode="after")
    def validate_component_composition(self) -> "ScenarioDefinition":
        if self.scenario_schema_version == "2.0":
            if self.video is not None or self.synthetic_bear_tags:
                raise ValueError("video and generated BearTag series require scenario schema 3.0")
            return self
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
