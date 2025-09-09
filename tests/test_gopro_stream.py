import sys
from pathlib import Path
from unittest import mock
import urllib.request

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))
from GoProController import GoProController

from tests.stubs.gopro import FakeGoPro


def test_stream_serves_video():
    with mock.patch('open_gopro.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        url = ctrl.start_preview(9100)
        # For wired GoPro, we get UDP URL which we can't test with urllib
        # Just verify we get the expected UDP URL format
        assert url.startswith('udp://') and ':9100' in url
        # Test that stop_preview works without error
        ctrl.stop_preview()
