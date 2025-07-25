import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
sys.path.append(str(MODULE_DIR))

import ble_beacon_handler as ble


def create_advertisement_bytes():
    data = bytearray()
    data.append(ble.KSENSOR_TYPE)
    data.extend((0x00, 0x09))
    data.extend((0x01, 0x2c))
    data.extend((0x00, 0x00))
    data.extend((0x03, 0xe8))
    data.extend((0x00, 0x00))
    data.extend((0xfc, 0x18))
    return bytes(data)


def test_rssi_to_distance(tmp_path):
    cfg = tmp_path / "rssi.ini"
    cfg.write_text("[RSSI_TO_DISTANCE]\nrssi_values=-80,-60,-40\ndistance_values=10,5,2\n")
    func = ble.load_rssi_distance_table(cfg)
    assert pytest.approx(ble.rssi_to_distance(-70, func), rel=1e-6) == 7.5
    assert pytest.approx(ble.rssi_to_distance(-90, func), rel=1e-6) == 12.5
    assert pytest.approx(ble.rssi_to_distance(-30, func), rel=1e-6) == 0.5


def test_acceleration_norm():
    handler = ble.BleBeaconHandler()
    _, acc = handler.decode_sensor_data(create_advertisement_bytes())
    assert pytest.approx(acc.norm, rel=1e-6) == 2 ** 0.5
    assert acc.is_moving is True


def test_full_flow(tmp_path):
    cfg = tmp_path / "rssi.ini"
    cfg.write_text("[RSSI_TO_DISTANCE]\nrssi_values=-80,-60,-40\ndistance_values=10,5,2\n")
    ble.RSSI_DISTANCE_FUNC = ble.load_rssi_distance_table(cfg)

    handler = ble.BleBeaconHandler()
    device = SimpleNamespace(address="AA", name="KBPro-test")
    adv = SimpleNamespace(rssi=-70, tx_power=-10, service_data={"0000": create_advertisement_bytes()})
    asyncio.run(handler.discovery_callback(device, adv))

    assert handler.advertisement_queue.qsize() == 1
    item = asyncio.run(handler.advertisement_queue.get())
    assert pytest.approx(item["distance"], rel=1e-6) == 7.5
    assert item["acc_sensor"].is_moving is True
