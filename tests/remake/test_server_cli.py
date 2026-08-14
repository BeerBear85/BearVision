import json
from pathlib import Path

from bearvision.server.cli import main


def write_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "server.yaml"
    path.write_text(
        """config_schema_version: "1.0"
config_kind: bearvision-server
admin: {host: 127.0.0.1, port: 4320}
registry_path: data/registry.json
scratch_dir: temp/scratch
local_queue_root: data/queue
""",
        encoding="utf-8",
    )
    return path


def test_admin_cli_uses_authoritative_registry_and_queue_snapshot(
    tmp_path: Path, capsys
) -> None:
    config = write_config(tmp_path)
    assert main(["--config", str(config), "create-user", "--email", " Bear@Example.com ", "--display-name", "Bear"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "bear@example.com"
    assert main(["--config", str(config), "create-tag", "--id", "BearTag-1"]) == 0
    capsys.readouterr()
    assert main(
        [
            "--config",
            str(config),
            "validate-assignment",
            "--user-id",
            "bear@example.com",
            "--bear-tag-id",
            "BearTag-1",
            "--valid-from",
            "2026-08-13T08:00:00Z",
            "--valid-to",
            "2026-08-13T09:00:00Z",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert main(["--config", str(config), "list-users"]) == 0
    assert json.loads(capsys.readouterr().out)["items"][0]["id"] == "bear@example.com"

    assert main(["--config", str(config), "snapshot"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["queue"]["counts"]["ready"] == 0
    assert snapshot["registry"]["users"][0]["id"] == "bear@example.com"
