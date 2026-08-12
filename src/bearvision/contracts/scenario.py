"""Versioned behavioural scenario contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


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


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_schema_version: Literal["2.0"]
    name: str = Field(min_length=1, max_length=150)
    seed: int = 0
    duration_s: float = Field(gt=0)
    timeline: tuple[TimelineEvent, ...]
    faults: ScenarioFaults = Field(default_factory=ScenarioFaults)
    expect: ScenarioExpectation = Field(default_factory=ScenarioExpectation)


def load_scenario(path: str | Path) -> ScenarioDefinition:
    """Load and strictly validate a YAML scenario."""

    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return ScenarioDefinition.model_validate(data)
