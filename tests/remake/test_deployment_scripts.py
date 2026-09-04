from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def script(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_linux_deployment_scripts_are_checked_out_with_lf() -> None:
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "*.sh text eol=lf" in attributes
    for manifest in (
        "pyproject.toml",
        "uv.lock",
        "apps/edge-control/package.json",
        "apps/edge-control/pnpm-lock.yaml",
    ):
        assert f"{manifest} text eol=lf" in attributes


def test_code_only_redeploy_runs_updater_without_root() -> None:
    source = script("redeploy-edge.ps1")

    assert "'bash scripts/update-raspberry-pi-code.sh'" in source
    assert "'sudo bash scripts/update-raspberry-pi-code.sh'" not in source
    assert "$ConfigureCodeDeploy" in source
    assert "sudo bash scripts/configure-code-deployment.sh" in source
    assert "[string[]]$payload = if ($ConfigureCodeDeploy)" in source
    assert '$remoteCommand = $remoteCommand.Replace("`r`n", "`n")' in source


def test_redeploy_stops_gopro_hindsight_before_running_deployment() -> None:
    source = script("redeploy-edge.ps1")

    assert "'scripts/stop_gopro_hindsight.py'" in source
    deployment = source[source.index("try {") :]
    assert deployment.index("Stop-GoProHindsightBeforeRedeploy") < deployment.index(
        "Creating Edge deployment archive"
    )


def test_code_updater_only_escalates_the_service_restart() -> None:
    source = script("update-raspberry-pi-code.sh")

    assert '[[ $EUID -ne 0 ]]' in source
    assert 'sudo -n /usr/bin/systemctl restart "$SERVICE_NAME.service"' in source
    assert "same_manifest_content()" in source
    assert "cmp --silent <(tr -d '\\r'" in source
    assert "runuser" not in source
    assert "chown -R" not in source


def test_full_setup_configures_least_privilege_code_deployment() -> None:
    source = script("setup-raspberry-pi.sh")

    assert "--deploy-user" in source
    assert "configure-code-deployment.sh" in source


def test_code_deployment_bootstrap_is_narrowly_scoped() -> None:
    source = script("configure-code-deployment.sh")

    assert "/etc/sudoers.d/bearvision-code-deploy" in source
    assert "NOPASSWD: /usr/bin/systemctl restart %s.service" in source
    assert '"$DEPLOY_USER" "$CONTROL_SERVICE_NAME"' in source
    assert 'usermod --append --groups "$SERVICE_GROUP" "$DEPLOY_USER"' in source
    assert '[[ $UNIT_USER == "$SERVICE_USER" ]]' in source


def test_full_setup_waits_for_bluetooth_controller_after_restart() -> None:
    source = script("setup-raspberry-pi.sh")

    assert "power_on_bluetooth()" in source
    assert "for attempt in {1..15}" in source
    assert "bluetoothctl power on" in source
    assert 'die "Bluetooth controller did not become ready' in source
