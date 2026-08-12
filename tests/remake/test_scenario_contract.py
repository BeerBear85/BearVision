from pathlib import Path

from bearvision.contracts import load_scenario


def test_example_scenario_is_versioned_and_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    scenario = load_scenario(root / "specs" / "scenarios" / "single-rider-success.yaml")

    assert scenario.scenario_schema_version == "1.0"
    assert scenario.timeline[0].event == "tag_enters_range"
