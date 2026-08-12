from pathlib import Path

from bearvision.contracts import load_scenario
from bearvision.simulation import VirtualClock, generate_bear_tag_series


ROOT = Path(__file__).resolve().parents[2]


def test_video_scenario_declares_each_component_source() -> None:
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")

    assert scenario.scenario_schema_version == "3.0"
    assert scenario.components.frames == "video"
    assert scenario.components.detector == "yolo"
    assert scenario.components.bear_tag == "synthetic"
    assert scenario.components.camera == "recorded_video"
    assert scenario.video is not None
    assert (ROOT / scenario.video.path).is_file()


def test_video_scenario_generates_deterministic_ten_hertz_bear_tag_data() -> None:
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")

    observations, registry = generate_bear_tag_series(
        scenario.synthetic_bear_tags,
        VirtualClock(),
    )

    assert len(observations) == 131
    assert registry[0].rider_id == "rider-video"
    active = [item for item in observations if 6.0 <= item.observed_at_monotonic_s <= 11.0]
    assert len(active) == 51
    assert all(item.rssi_dbm == -50 for item in active)
    assert all(item.acceleration_mps2.z == 19 for item in active)
