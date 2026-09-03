from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def script(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_code_only_redeploy_runs_updater_without_root() -> None:
    source = script("redeploy-edge.ps1")

    assert "'bash scripts/update-raspberry-pi-code.sh'" in source
    assert "'sudo bash scripts/update-raspberry-pi-code.sh'" not in source
    assert "$ConfigureCodeDeploy" in source
    assert "sudo bash scripts/configure-code-deployment.sh" in source
    assert "[string[]]$payload = if ($ConfigureCodeDeploy)" in source


def test_code_updater_only_escalates_the_service_restart() -> None:
    source = script("update-raspberry-pi-code.sh")

    assert '[[ $EUID -ne 0 ]]' in source
    assert 'sudo -n /usr/bin/systemctl restart "$SERVICE_NAME.service"' in source
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
