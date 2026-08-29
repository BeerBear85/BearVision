import json
from io import StringIO
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


def invoke(monkeypatch, config: Path, command: str, **parameters) -> int:
    envelope = {
        "commandSchemaVersion": "1.0",
        "command": command,
        **parameters,
    }
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(envelope)))
    return main(["--config", str(config), "execute"])


def test_admin_cli_uses_authoritative_registry_and_queue_snapshot(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    config = write_config(tmp_path)
    assert invoke(
        monkeypatch,
        config,
        "create-user",
        email=" Bear@Example.com ",
        displayName="Bear",
    ) == 0
    user = json.loads(capsys.readouterr().out)
    user_id = user["id"]
    assert user["email"] == "bear@example.com"
    assert invoke(monkeypatch, config, "create-tag", id="BearTag-1") == 0
    capsys.readouterr()
    assignment = {
        "userId": user_id,
        "bearTagId": "BearTag-1",
        "validFrom": "2026-08-13T08:00:00Z",
        "validTo": "2026-08-13T09:00:00Z",
    }
    assert invoke(
        monkeypatch,
        config,
        "validate-assignment",
        **assignment,
    ) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert invoke(
        monkeypatch,
        config,
        "create-assignment",
        id="assignment-1",
        **assignment,
    ) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "assignment-1"

    assert invoke(monkeypatch, config, "list-users") == 0
    assert json.loads(capsys.readouterr().out)["items"][0]["id"] == user_id

    assert invoke(
        monkeypatch,
        config,
        "update-user-email",
        userId=user_id,
        email="new-bear@example.com",
    ) == 0
    updated_user = json.loads(capsys.readouterr().out)
    assert updated_user["id"] == user_id
    assert updated_user["email"] == "new-bear@example.com"

    assert invoke(
        monkeypatch,
        config,
        "list-user-videos",
        userId=" new-bear@example.com ",
    ) == 0
    user_videos = json.loads(capsys.readouterr().out)
    assert user_videos["user"]["email"] == "new-bear@example.com"
    assert user_videos["items"] == []

    assert invoke(monkeypatch, config, "list-tags") == 0
    assert json.loads(capsys.readouterr().out)["items"][0]["id"] == "BearTag-1"

    assert invoke(monkeypatch, config, "summary") == 0
    assert json.loads(capsys.readouterr().out)["counts"]["ready"] == 0

    assert invoke(monkeypatch, config, "list-jobs", status="ready") == 0
    assert json.loads(capsys.readouterr().out)["total"] == 0

    assert invoke(monkeypatch, config, "requeue", jobId="missing") == 0
    assert json.loads(capsys.readouterr().out) == {"requeued": False}

    assert invoke(monkeypatch, config, "run-once") == 0
    assert json.loads(capsys.readouterr().out) is None

    assert invoke(monkeypatch, config, "snapshot") == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["queue"]["counts"]["ready"] == 0
    assert snapshot["registry"]["users"][0]["id"] == user_id

    assert invoke(monkeypatch, config, "job-detail", jobId="missing") == 1
    assert "missing" in json.loads(capsys.readouterr().err)["error"]


def test_admin_cli_rejects_unknown_envelope_fields(tmp_path: Path, capsys, monkeypatch) -> None:
    config = write_config(tmp_path)

    assert invoke(monkeypatch, config, "summary", unexpectedPolicy="node-owned") == 1

    error = json.loads(capsys.readouterr().err)["error"]
    assert "unexpectedPolicy" in error
    assert "Extra inputs are not permitted" in error
