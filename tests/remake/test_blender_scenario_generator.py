import json
from pathlib import Path

import pytest

from bearvision.contracts import load_scenario
from bearvision.simulation import VirtualClock, generate_bear_tag_series
from bearvision.simulation.blender_scenario import (
    generate_blender_scenario,
    simulated_rssi_dbm,
    write_scenario,
)


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "test/blender_scenes/wakeboard_fs360_60fps"
TWO_RIDER_SCENE = ROOT / "test/blender_scenes/wakeboard_two_riders_fs360_60fps"
GENERATED = ROOT / "specs/scenarios/wakeboard-fs360-60fps-blender-regression.yaml"
TWO_RIDER_GENERATED = (
    ROOT / "specs/scenarios/wakeboard-two-riders-60fps-blender-regression.yaml"
)


def _copy_scene_metadata(source: Path, target: Path) -> None:
    target.mkdir(parents=True)
    motions = sorted(source.glob("*_rider[0-9]*_motion.json"))
    for metadata in [*motions, *source.glob("*_camera_info.yaml")]:
        (target / metadata.name).write_bytes(metadata.read_bytes())
    source_clip = json.loads(motions[0].read_text(encoding="utf-8"))["source"]["clip"]
    (target / source_clip).write_bytes(b"local-video-placeholder")


@pytest.fixture
def scene_dir(tmp_path: Path) -> Path:
    """Mirror tracked scene metadata and add an untracked local video placeholder."""

    target = tmp_path / SCENE.relative_to(ROOT)
    _copy_scene_metadata(SCENE, target)
    return target


@pytest.fixture
def two_rider_scene_dir(tmp_path: Path) -> Path:
    target = tmp_path / TWO_RIDER_SCENE.relative_to(ROOT)
    _copy_scene_metadata(TWO_RIDER_SCENE, target)
    return target


def test_log_distance_rssi_is_deterministic_and_bounded() -> None:
    assert simulated_rssi_dbm(1) == -50
    assert simulated_rssi_dbm(10) == -70
    assert simulated_rssi_dbm(100) == -90
    assert simulated_rssi_dbm(1e9) == -127
    with pytest.raises(ValueError, match="distance_m"):
        simulated_rssi_dbm(0)


def test_generator_converts_blender_frames_into_bear_tag_observations(
    scene_dir: Path,
    tmp_path: Path,
) -> None:
    scenario = generate_blender_scenario(scene_dir, repository_root=tmp_path)
    series = scenario.synthetic_bear_tags[0]
    samples = series.samples

    assert scenario.scenario_schema_version == "3.1"
    assert scenario.generated_from is not None
    assert scenario.video is not None
    assert series.tag_id == "bear_tag_666"
    assert len(samples) == 96
    assert samples[0].source_frame == 1
    assert samples[-1].source_frame == 571
    assert samples[0].source_distance_m == pytest.approx(33.903526)
    assert min(sample.source_distance_m for sample in samples if sample.source_distance_m) \
        == pytest.approx(26.800187)
    assert samples[0].rssi_dbm == -81
    # Specific force is Blender world acceleration minus world gravity.
    assert samples[0].acceleration_mps2.z == pytest.approx(-4.208 + 9.80665)

    observations, registry = generate_bear_tag_series(
        scenario.synthetic_bear_tags,
        VirtualClock(),
    )
    assert len(observations) == len(samples)
    assert observations[0].battery_voltage_mv == 3000
    assert registry[0].tag_id == "bear_tag_666"
    assert registry[0].rider_id == "rider-wakeboard-fs360-60fps"


def test_generator_preserves_two_rider_bear_tags_and_video_timing(
    two_rider_scene_dir: Path,
    tmp_path: Path,
) -> None:
    scenario = generate_blender_scenario(two_rider_scene_dir, repository_root=tmp_path)
    rider1, rider2 = scenario.synthetic_bear_tags

    assert scenario.duration_s == pytest.approx(19.567)
    assert (rider1.tag_id, rider2.tag_id) == ("bear_tag_666", "bear_tag_123")
    assert rider1.rider_id.endswith("-rider1")
    assert rider2.rider_id.endswith("-rider2")
    assert rider1.start_s == 0
    assert rider2.start_s == pytest.approx(379 / 60)
    assert rider1.samples[0].source_frame == 1
    assert rider2.samples[0].source_frame == 380
    assert rider2.samples[0].at_s == pytest.approx(379 / 60)
    assert scenario.generated_from is not None
    assert scenario.generated_from.motion_path is None
    assert len(scenario.generated_from.motion_paths) == 2


def test_checked_in_scenario_is_reproducible_and_writer_is_safe(
    scene_dir: Path,
    tmp_path: Path,
) -> None:
    expected = generate_blender_scenario(scene_dir, repository_root=tmp_path)
    assert load_scenario(GENERATED) == expected

    output = tmp_path / "scenario.yaml"
    write_scenario(expected, output)
    assert load_scenario(output) == expected
    with pytest.raises(FileExistsError):
        write_scenario(expected, output)


def test_checked_in_two_rider_scenario_is_reproducible(
    two_rider_scene_dir: Path,
    tmp_path: Path,
) -> None:
    expected = generate_blender_scenario(two_rider_scene_dir, repository_root=tmp_path)
    assert load_scenario(TWO_RIDER_GENERATED) == expected


def test_generator_rejects_missing_camera_metadata(
    scene_dir: Path,
    tmp_path: Path,
) -> None:
    camera = scene_dir / "wakeboard_fs360_60fps_camera_info.yaml"
    camera.unlink()

    with pytest.raises(ValueError, match="camera metadata"):
        generate_blender_scenario(scene_dir, repository_root=tmp_path)
