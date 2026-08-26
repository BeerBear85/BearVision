from pathlib import Path

import pytest

from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system
from bearvision.simulation import runner as synthetic_runner_module
from bearvision.simulation import video_runner as video_runner_module


ROOT = Path(__file__).resolve().parents[2]


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
    ("field", "value"),
    [
        ("detector", "yolo"),
        ("bear_tag", "ble"),
        ("camera", "simulated_gopro"),
        ("storage", "box"),
    ],
)
def test_unsupported_synthetic_component_combinations_fail_explicitly(
    field: str,
    value: str,
) -> None:
    scenario = load_scenario(ROOT / "specs/scenarios/single-rider-success.yaml")
    components = scenario.components.model_copy(update={field: value})
    unsupported = scenario.model_copy(update={"components": components})

    with pytest.raises(ValueError, match="declared but not implemented"):
        build_behavioral_system(unsupported)


def test_unsupported_video_component_combination_is_rejected_by_video_runner(
    monkeypatch,
) -> None:
    class VideoRunner:
        @classmethod
        def from_scenario(cls, scenario, **kwargs):
            if scenario.components.storage != "memory":
                raise ValueError("video regression currently requires storage=memory")
            raise AssertionError("test requires an unsupported combination")

    monkeypatch.setattr(video_runner_module, "VideoScenarioRunner", VideoRunner)
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")
    components = scenario.components.model_copy(update={"storage": "box"})
    unsupported = scenario.model_copy(update={"components": components})

    with pytest.raises(ValueError, match="requires storage=memory"):
        build_behavioral_system(unsupported)
