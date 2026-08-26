from pathlib import Path

from bearvision.contracts import ScenarioComponents, load_scenario
from bearvision.processing import VirtualCameramanJobProcessor
from bearvision.simulation import VirtualClock, generate_bear_tag_series
from bearvision.simulation.video_runner import VideoScenarioRunner


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_recorded_video_camera_is_migrated_to_gopro_emulator() -> None:
    components = ScenarioComponents.model_validate(
        {
            "frames": "video",
            "detector": "yolo",
            "bear_tag": "synthetic",
            "camera": "recorded_video",
            "storage": "memory",
        }
    )

    assert components.camera == "simulated_gopro"


def test_every_video_scenario_uses_the_gopro_emulator() -> None:
    video_scenarios = []
    for path in sorted((ROOT / "specs/scenarios").glob("*.yaml")):
        scenario = load_scenario(path)
        if scenario.components.frames == "video":
            video_scenarios.append(path.name)
            assert scenario.components.camera == "simulated_gopro", path.name

    assert len(video_scenarios) == 6


def test_video_scenario_declares_each_component_source() -> None:
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")

    assert scenario.scenario_schema_version == "3.0"
    assert scenario.components.frames == "video"
    assert scenario.components.detector == "yolo"
    assert scenario.components.bear_tag == "synthetic"
    assert scenario.components.camera == "simulated_gopro"
    assert scenario.video is not None
    assert (ROOT / scenario.video.path).is_file()


def test_video_scenario_generates_deterministic_ten_hertz_bear_tag_data() -> None:
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")

    observations, registry = generate_bear_tag_series(
        scenario.synthetic_bear_tags,
        VirtualClock(),
    )

    assert len(observations) == 153
    assert registry[0].tag_id == "bear_tag_666"
    assert registry[0].rider_id == "rider-video"
    active = [item for item in observations if 6.0 <= item.observed_at_monotonic_s <= 11.0]
    assert len(active) == 51
    assert all(item.rssi_dbm == -72 for item in active)
    assert all(item.acceleration_mps2.z == 18.8 for item in active)


def test_video_runner_uses_edge_processing_and_does_not_start_server_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeDnnHandler:
        confidence_threshold = 0.0

        def __init__(self, model: str) -> None:
            self.model = model

        def init(self) -> None:
            pass

    def unexpected_server_worker(*args, **kwargs):
        raise AssertionError("process_server=False started the server worker")

    monkeypatch.setattr(
        "bearvision.integrations.opencv_dnn.DnnHandler",
        FakeDnnHandler,
    )
    monkeypatch.setattr(
        "bearvision.simulation.video_runner.ServerWorker",
        unexpected_server_worker,
    )
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")

    runner = VideoScenarioRunner.from_scenario(
        scenario,
        repository_root=ROOT,
        capture_dir=tmp_path / "captures",
        process_server=False,
    )

    assert runner.worker is None
    assert runner.orchestrator.upload_enabled
    assert isinstance(runner.orchestrator.clip_processor, VirtualCameramanJobProcessor)
