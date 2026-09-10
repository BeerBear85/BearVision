from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bearvision.edge.gopro_diagnostics import diagnose_gopro


class _Connection:
    def close(self) -> None:
        pass


class _Response:
    status = 200

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def read(self, _limit: int) -> bytes:
        return json.dumps({"status": {}}).encode()


def _usb_device(root: Path) -> None:
    device = root / "1-1"
    device.mkdir(parents=True)
    (device / "idVendor").write_text("2672\n", encoding="utf-8")
    (device / "manufacturer").write_text("GoPro\n", encoding="utf-8")
    (device / "product").write_text("HERO\n", encoding="utf-8")


def _network_result(*, address: str = "172.24.106.2", prefix: int = 24) -> SimpleNamespace:
    return SimpleNamespace(
        returncode=0,
        stderr="",
        stdout=json.dumps([{
            "ifname": "eth1",
            "addr_info": [{"family": "inet", "local": address, "prefixlen": prefix}],
        }]),
    )


def test_diagnostics_reports_every_successful_connection_layer(tmp_path: Path) -> None:
    _usb_device(tmp_path)

    report = diagnose_gopro(
        sysfs_root=tmp_path,
        command_runner=lambda _args: _network_result(),
        tcp_connect=lambda *_args, **_kwargs: _Connection(),
        open_url=lambda *_args, **_kwargs: _Response(),
    )

    assert report.status == "pass"
    assert [check.status for check in report.checks] == ["pass", "pass", "pass", "pass"]
    assert "communicate" in report.summary


def test_diagnostics_identifies_usb_as_the_first_failed_layer(tmp_path: Path) -> None:
    def connection_failed(*_args: object, **_kwargs: object) -> object:
        raise OSError("host unreachable")

    report = diagnose_gopro(
        sysfs_root=tmp_path,
        command_runner=lambda _args: _network_result(address="192.168.1.20"),
        tcp_connect=connection_failed,
        open_url=connection_failed,
    )

    assert report.status == "fail"
    assert report.checks[0].check_id == "usb_device"
    assert report.checks[0].status == "fail"
    assert "cannot currently detect" in report.summary
    assert "data-capable cable" in (report.checks[0].corrective_action or "")
