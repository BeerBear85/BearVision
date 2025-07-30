import sys
from pathlib import Path
from unittest import mock

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))
from GoProController import GoProController

from tests.stubs.gopro import FakeGoPro
from open_gopro.models.constants import constants


def test_list_and_download(tmp_path):
    with mock.patch('GoProController.WirelessGoPro', FakeGoPro):
        ctrl = GoProController()
        files = ctrl.list_videos()
        assert files == ['DCIM/100GOPRO/GOPR0001.MP4']

        out = tmp_path / 'f.mp4'
        ctrl.download_file('DCIM/100GOPRO/GOPR0001.MP4', str(out))
        assert out.exists()


def test_configure_and_preview():
    with mock.patch('GoProController.WirelessGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.configure()
        gopro = ctrl._gopro
        assert gopro.http_command.group is not None
        assert gopro.http_settings.hindsight.value is not None
        url = ctrl.start_preview(9000)
        assert url == 'udp://127.0.0.1:9000'


def test_start_hindsight_clip():
    with mock.patch('GoProController.WirelessGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.start_hindsight_clip(0)
        gopro = ctrl._gopro
        assert gopro.http_command.shutter == [constants.Toggle.ENABLE, constants.Toggle.DISABLE]

