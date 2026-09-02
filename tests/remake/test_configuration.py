from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from bearvision.config import load_edge_config, load_server_config
from bearvision.config.models import AssignmentConfig
from bearvision.config.migrate import add_version_header, migrate_edge_data, write_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_active_edge_config_is_versioned_and_valid() -> None:
    config = load_edge_config(REPO_ROOT / "config" / "edge.yaml")
    assert config.config_schema_version == "3.0"
    assert config.storage.provider == "box"
    assert not hasattr(config, "assignment")
    assert config.clip_extraction.engine == "ffmpeg"
    assert config.recording.hindsight_duration_s == 15
    assert config.clip_extraction.crf == 20
    assert config.virtual_cameraman.crop_width_ratio == 0.5
    assert config.virtual_cameraman.output_width_px == 960
    assert config.virtual_cameraman.output_height_px == 540
    assert config.virtual_cameraman.output_crf == 18
    assert config.error_recovery.max_restarts == 0
    assert config.readiness.camera_preview_timeout_s == 12
    assert config.readiness.ble_scan_duration_s == 2
    assert config.readiness.cleanup_timeout_s == 3


def test_assignment_fusion_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        AssignmentConfig(motion_weight=0.8, rssi_weight=0.3)


def test_server_owns_assignment_policy_and_local_admin_binding() -> None:
    config = load_server_config(REPO_ROOT / "config" / "server.yaml")
    assert config.assignment.motion_weight == 0.7
    assert config.admin.host == "127.0.0.1"


def test_all_active_configs_start_with_independent_version_header() -> None:
    expected_versions = {
        "edge.yaml": "3.0",
        "server.yaml": "1.0",
        "ble-test.yaml": "2.0",
        "training.yaml": "2.0",
        "annotation-example.yaml": "2.0",
    }
    for name, version in expected_versions.items():
        lines = (REPO_ROOT / "config" / name).read_text(encoding="utf-8").splitlines()
        assert lines[0] == f'config_schema_version: "{version}"'
        assert lines[1].startswith("config_kind: bearvision-")


def test_missing_or_unsupported_config_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "edge.yaml"
    path.write_text("config_kind: bearvision-edge\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_edge_config(path)

    path.write_text(
        'config_schema_version: "9.0"\nconfig_kind: bearvision-edge\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_edge_config(path)


def test_legacy_edge_files_migrate_with_yaml_precedence(tmp_path: Path) -> None:
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[EDGE_APPLICATION]\nrecording_duration=30\nstream_max_lag_ms=1000\n"
        "[STORAGE_COMMON]\nsecret_key_name=PRIMARY\n"
        "[BOX]\nroot_folder=clips\n"
        "[ANNOTATION_GUI]\npreview_width=280\n",
        encoding="utf-8",
    )
    legacy_yaml = tmp_path / "edge.yaml"
    legacy_yaml.write_text(
        "recording:\n  post_detection_duration: 5\n",
        encoding="utf-8",
    )

    migrated, warnings = migrate_edge_data(ini, legacy_yaml)

    assert migrated["config_schema_version"] == "3.0"
    assert migrated["recording"]["post_detection_duration_s"] == 5
    assert migrated["storage"]["root_folder"] == "clips"
    assert any("ANNOTATION_GUI" in warning for warning in warnings)


def test_migration_never_overwrites_target(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_yaml(target, {"config_schema_version": "2.0"})


def test_unversioned_yaml_gets_header_first() -> None:
    migrated = add_version_header({"epochs": 50}, kind="bearvision-training")
    dumped = yaml.safe_dump(migrated, sort_keys=False)
    assert dumped.splitlines()[:2] == [
        "config_schema_version: '2.0'",
        "config_kind: bearvision-training",
    ]
