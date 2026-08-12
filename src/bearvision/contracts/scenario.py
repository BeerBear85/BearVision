"""Versioned behavioural scenario contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class TimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    at_s: float = Field(ge=0)
    event: str = Field(min_length=1, max_length=100)
    payload: dict[str, Any] = Field(default_factory=dict)


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_schema_version: Literal["1.0"]
    name: str = Field(min_length=1, max_length=150)
    seed: int = 0
    duration_s: float = Field(gt=0)
    timeline: tuple[TimelineEvent, ...]
    faults: dict[str, Any] = Field(default_factory=dict)
    expect: dict[str, Any] = Field(default_factory=dict)


def load_scenario(path: str | Path) -> ScenarioDefinition:
    """Load and strictly validate a YAML scenario."""

    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return ScenarioDefinition.model_validate(data)
