import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
sys.path.append(str(MODULE_DIR))

from ble_beacon_handler import BleBeaconHandler, KSENSOR_TYPE


def create_advertisement_bytes():
    # frame type
    data = bytearray()
    data.append(KSENSOR_TYPE)
    # sensor mask: voltage + acceleration
    data.extend((0x00, 0x09))
    # battery level 0x012c -> 300
    data.extend((0x01, 0x2c))
    # two padding bytes
    data.extend((0x00, 0x00))
    # acceleration x=1000, y=0, z=-1000
    data.extend((0x03, 0xe8))  # x
    data.extend((0x00, 0x00))  # y
    data.extend((0xfc, 0x18))  # z
    return bytes(data)


def test_decode_sensor_data():
    handler = BleBeaconHandler()
    battery, acc = handler.decode_sensor_data(create_advertisement_bytes())
    assert battery == 300
    assert pytest.approx(acc.x, rel=1e-5) == 1.0
    assert pytest.approx(acc.y, rel=1e-5) == 0.0
    assert pytest.approx(acc.z, rel=1e-5) == -1.0


def test_discovery_callback_parses_advertisement():
    data = create_advertisement_bytes()
    handler = BleBeaconHandler()
    device = SimpleNamespace(address="AA", name="KBPro-test")
    adv = SimpleNamespace(rssi=-42, tx_power=-10, service_data={"0000": data})
    asyncio.run(handler.discovery_callback(device, adv))

    assert handler.advertisement_queue.qsize() == 1
    item = asyncio.run(handler.advertisement_queue.get())
    assert item["address"] == "AA"
    assert item["name"] == "KBPro-test"
    assert item["rssi"] == -42
    assert item["tx_power"] == -10
    acc = item["acc_sensor"]
    assert pytest.approx(acc.x, rel=1e-5) == 1.0
    assert pytest.approx(acc.z, rel=1e-5) == -1.0
