import sys
from pathlib import Path
from unittest import mock
import tempfile
import yaml

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from tests.stubs.gopro import FakeGoPro
from open_gopro.models.constants import constants

# Import GoProController after setting up the mock
from GoProController import GoProController
from GoProConfig import GoProConfiguration


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
        # Verify that startHindsightMode sets hindsight to 15 seconds instead of triggering recording
        assert gopro.http_settings.hindsight.value == 2  # 15 seconds (NUM_15_SECONDS)
        # Verify that no shutter commands were triggered
        assert gopro.http_command.shutter == []
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
        assert 'status' in status
        assert 'recording' in status['status']
        
        ctrl.disconnect()


def test_download_configuration():
    """Test downloading configuration from GoPro."""
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()
        
        # Create a temporary file for the configuration
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock the camera state response
            with mock.patch.object(ctrl._gopro.http_command, 'get_camera_state') as mock_get_state:
                mock_state_data = {
                    'settings': {
                        'video_resolution': 12,  # 4K
                        'frames_per_second': 8,  # 60 FPS
                        'hindsight': 1  # 15 seconds
                    },
                    'status': {'encoding': False}
                }
                
                # Create a mock response object
                class MockResponse:
                    def __init__(self, data):
                        self.data = data
                
                mock_get_state.return_value = MockResponse(mock_state_data)
                
                # Download configuration
                saved_path = ctrl.download_configuration(temp_path)
                
                assert saved_path == temp_path
                assert Path(temp_path).exists()
                
                # Verify YAML content
                with open(temp_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                assert isinstance(config_data, dict)
                assert 'config_name' in config_data
                assert 'config_version' in config_data
                
        finally:
            Path(temp_path).unlink(missing_ok=True)
            ctrl.disconnect()


def test_download_configuration_no_connection():
    """Test downloading configuration without GoPro connection."""
    ctrl = GoProController()
    
    # Test without connecting first
    try:
        ctrl.download_configuration()
        assert False, "Should have raised ConnectionError"
    except ConnectionError as e:
        # Accept either connection error message
        assert any(msg in str(e) for msg in ["No GoPro connection available", "GoPro connection lost"])


def test_upload_configuration():
    """Test uploading configuration to GoPro."""
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()
        
        # Create a test configuration file
        test_config = GoProConfiguration(
            video_resolution="4K",
            frame_rate="60",
            hindsight="15 seconds",
            config_name="Test Upload Config"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config.to_dict(), f)
            temp_path = f.name
        
        try:
            # Mock the _apply_configuration method to avoid actual GoPro calls
            with mock.patch.object(ctrl, '_run_in_thread') as mock_run:
                result = ctrl.upload_configuration(temp_path)
                
                assert result is True
                mock_run.assert_called_once()
                
        finally:
            Path(temp_path).unlink(missing_ok=True)
            ctrl.disconnect()


def test_upload_configuration_invalid_file():
    """Test uploading configuration from invalid file."""
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        ctrl.connect()
        
        # Test with non-existent file
        try:
            ctrl.upload_configuration("nonexistent_file.yaml")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test with invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            ctrl.upload_configuration(temp_path)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        finally:
            Path(temp_path).unlink(missing_ok=True)
            ctrl.disconnect()


def test_upload_configuration_no_connection():
    """Test uploading configuration without GoPro connection."""
    # Create a valid config file
    test_config = GoProConfiguration(config_name="Test No Connection")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config.to_dict(), f)
        temp_path = f.name
    
    try:
        ctrl = GoProController()
        # Test without connecting first
        ctrl.upload_configuration(temp_path)
        assert False, "Should have raised ConnectionError"
    except ConnectionError as e:
        # Accept either connection error message
        assert any(msg in str(e) for msg in ["No GoPro connection available", "GoPro connection lost"])
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_validate_configuration():
    """Test configuration file validation."""
    with mock.patch('GoProController.WiredGoPro', FakeGoPro):
        ctrl = GoProController()
        
        # Test with valid configuration
        valid_config = GoProConfiguration(config_name="Valid Test Config")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config.to_dict(), f)
            temp_path = f.name
        
        try:
            is_valid, message = ctrl.validate_configuration(temp_path)
            assert is_valid is True
            assert "valid" in message.lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
        
        # Test with invalid configuration
        invalid_config = {
            "video_resolution": "Invalid Resolution",
            "frame_rate": "999"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            is_valid, message = ctrl.validate_configuration(temp_path)
            assert is_valid is False
            assert len(message) > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

