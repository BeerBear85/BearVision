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
GENERATED = ROOT / "specs/scenarios/wakeboard-fs360-60fps-blender-regression.yaml"


@pytest.fixture
def scene_dir(tmp_path: Path) -> Path:
    """Mirror tracked scene metadata and add an untracked local video placeholder."""

    target = tmp_path / SCENE.relative_to(ROOT)
    target.mkdir(parents=True)
    motion = next(SCENE.glob("*_rider_motion.json"))
    (target / motion.name).write_bytes(motion.read_bytes())
    (target / "wakeboard_fs360_60fps.mp4").write_bytes(b"local-video-placeholder")
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
    assert len(samples) == 51
    assert samples[0].source_frame == 1
    assert samples[-1].source_frame == 301
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
    assert registry[0].rider_id == "rider-wakeboard-fs360-60fps"


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


def test_generator_rejects_camera_metadata_disagreement(
    scene_dir: Path,
    tmp_path: Path,
) -> None:
    camera = scene_dir / "wakeboard_fs360_60fps_camera_info.yaml"
    camera.write_text(
        "camera:\n"
        "  static: true\n"
        "  transform_world:\n"
        "    location_m:\n"
        "      x: -9.0\n"
        "      y: -30.5\n"
        "      z: 3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagree"):
        generate_blender_scenario(scene_dir, repository_root=tmp_path)
