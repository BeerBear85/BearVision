import csv
from collections import deque
from pathlib import Path

import pytest


def test_ble_ringbuffer_loads_simulated_data():
    data_file = Path(__file__).parent.parent / "data" / "ble_kicker_sim.csv"
    ringbuffer = deque(maxlen=300)

    with data_file.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entry = {k: float(v) for k, v in row.items()}
            ringbuffer.append(entry)

    assert len(ringbuffer) == 300
    first = ringbuffer[0]
    last = ringbuffer[-1]
    assert first["timestamp"] == 0.0
    assert pytest.approx(last["timestamp"], rel=1e-6) == 29.9

    # ensure kicker signature around midpoint
    data = list(ringbuffer)
    assert pytest.approx(data[150]["acc_z"], rel=1e-6) == 4.0
    assert pytest.approx(data[151]["acc_z"], rel=1e-6) == 0.0
    assert pytest.approx(data[152]["acc_z"], rel=1e-6) == 5.0

    # RSSI increases and peaks during jump
    rssis = [d["rssi"] for d in data]
    assert rssis[0] < rssis[150] < rssis[152]
    assert max(rssis) == rssis[152]
    assert rssis[152] > rssis[-1]
