import sys
from pathlib import Path
from unittest import mock

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from tests.stubs.gopro import FakeGoPro
from open_gopro.models.constants import constants

# Import GoProController after setting up the mock
from GoProController import GoProController


def test_list_and_download(tmp_path):
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()  # Connect first to set up the GoPro properly
        files = ctrl.list_videos()
        assert files == ['DCIM/100GOPRO/GOPR0001.MP4']

        out = tmp_path / 'f.mp4'
        ctrl.download_file('DCIM/100GOPRO/GOPR0001.MP4', str(out))
        assert out.exists()
        ctrl.disconnect()


def test_configure_and_preview():
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()  # Connect first to set up the GoPro properly
        ctrl.configure()
        gopro = ctrl._gopro
        assert gopro.http_command.group is not None
        assert gopro.http_settings.hindsight.value is not None
        url = ctrl.start_preview(9000)
        # For wired GoPro, expect UDP stream URL format
        assert url.startswith('udp://') and ':9000' in url
        ctrl.stop_preview()
        ctrl.disconnect()


def test_start_hindsight_clip():
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()  # Connect first to set up the GoPro properly
        ctrl.start_hindsight_clip(0)
        gopro = ctrl._gopro
        assert gopro.http_command.shutter == [constants.Toggle.ENABLE, constants.Toggle.DISABLE]
        ctrl.disconnect()


def test_startHindsightMode():
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()  # Connect first to set up the GoPro properly
        ctrl.startHindsightMode()
        gopro = ctrl._gopro
        # Verify that startHindsightMode triggers the same shutter sequence as start_hindsight_clip
        assert gopro.http_command.shutter == [constants.Toggle.ENABLE, constants.Toggle.DISABLE]
        ctrl.disconnect()


def test_recording_controls():
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()  # Connect first to set up the GoPro properly
        gopro = ctrl._gopro
        
        # Test start recording
        ctrl.start_recording()
        assert constants.Toggle.ENABLE in gopro.http_command.shutter
        
        # Reset shutter list for stop test
        gopro.http_command.shutter.clear()
        
        # Test stop recording  
        ctrl.stop_recording()
        assert gopro.http_command.shutter == [constants.Toggle.DISABLE]
        
        # Test get camera status
        status = ctrl.get_camera_status()
        assert isinstance(status, dict)
        assert 'recording' in status
        
        ctrl.disconnect()

