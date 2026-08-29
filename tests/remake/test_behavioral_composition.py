from pathlib import Path

import pytest
from pydantic import ValidationError

import bearvision.edge as edge_package
from bearvision.contracts import ScenarioDefinition, load_scenario
from bearvision.simulation import build_behavioral_system
from bearvision.simulation import runner as synthetic_runner_module
from bearvision.simulation import video_runner as video_runner_module


ROOT = Path(__file__).resolve().parents[2]


def test_behavioral_composition_is_not_exposed_by_the_edge_package() -> None:
    assert not hasattr(edge_package, "build_behavioral_system")


def test_supported_component_combinations_route_to_their_distinct_runners(
    monkeypatch,
) -> None:
    calls = []
    synthetic_result = object()
    video_result = object()

    class SyntheticRunner:
        @classmethod
        def from_scenario(cls, scenario, **kwargs):
            calls.append(("synthetic", scenario.name, kwargs))
            return synthetic_result

    class VideoRunner:
        @classmethod
        def from_scenario(cls, scenario, **kwargs):
            calls.append(("video", scenario.name, kwargs))
            return video_result

    monkeypatch.setattr(synthetic_runner_module, "ClosedLoopScenarioRunner", SyntheticRunner)
    monkeypatch.setattr(video_runner_module, "VideoScenarioRunner", VideoRunner)
    synthetic = load_scenario(ROOT / "specs/scenarios/single-rider-success.yaml")
    video = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")

    assert build_behavioral_system(synthetic, process_server=False) is synthetic_result
    assert build_behavioral_system(video, process_server=False) is video_result
    assert [(kind, name) for kind, name, _ in calls] == [
        ("synthetic", synthetic.name),
        ("video", video.name),
    ]
    assert all(call[2]["process_server"] is False for call in calls)


@pytest.mark.parametrize(
    ("scenario_name", "field", "value"),
    [
        ("single-rider-success.yaml", "detector", "yolo"),
        ("single-rider-success.yaml", "bear_tag", "ble"),
        ("single-rider-success.yaml", "camera", "simulated_gopro"),
        ("single-rider-success.yaml", "storage", "box"),
        ("wakeboard-video-yolo.yaml", "storage", "box"),
    ],
)
def test_unsupported_component_profiles_fail_during_scenario_validation(
    scenario_name: str,
    field: str,
    value: str,
) -> None:
    scenario = load_scenario(ROOT / "specs/scenarios" / scenario_name)
    raw = scenario.model_dump(mode="json")
    raw["components"][field] = value

    with pytest.raises(ValidationError, match="declared but not executable"):
        ScenarioDefinition.model_validate(raw)
