import sys
from pathlib import Path
from unittest import mock
import urllib.request

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))
from GoProController import GoProController

from tests.stubs.gopro import FakeGoPro


def test_stream_serves_video():
    with mock.patch('GoProController.WirelessGoPro', FakeGoPro):
        ctrl = GoProController()
        url = ctrl.start_preview(9100)
        import time
        time.sleep(0.1)
        with urllib.request.urlopen(url) as resp:
            data = resp.read(1024)
        assert data, 'no data returned from stream'
        ctrl._gopro.streaming.stop()
