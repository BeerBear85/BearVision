"""
Integration tests for GoPro configuration flow.

Tests the complete end-to-end workflow for:
- GUI → GoProController → GoProConfig → YAML files
- Full download and upload configuration flow
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch
import sys

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

# Add tools path for GUI
TOOLS_DIR = Path(__file__).resolve().parents[1] / 'tools'
sys.path.append(str(TOOLS_DIR))

from tests.stubs.gopro import FakeGoPro
from GoProController import GoProController
from GoProConfig import GoProConfiguration, VideoResolution, FrameRate, HindsightOption


class TestGoProConfigurationIntegration:
    """Integration tests for GoPro configuration workflow."""
    
    def test_complete_download_workflow(self):
        """Test complete configuration download workflow."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            # Initialize controller and connect
            ctrl = GoProController()
            ctrl.connect()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "test_config.yaml"
                
                # Download configuration
                saved_path = ctrl.download_configuration(str(config_path))
                
                # Verify file was created
                assert Path(saved_path).exists()
                assert saved_path == str(config_path)
                
                # Verify file contents
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                assert isinstance(config_data, dict)
                assert 'config_name' in config_data
                assert 'config_version' in config_data
                assert 'video_resolution' in config_data
                assert 'frame_rate' in config_data
                
            ctrl.disconnect()
    
    def test_complete_upload_workflow(self):
        """Test complete configuration upload workflow."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            # Initialize controller and connect
            ctrl = GoProController()
            ctrl.connect()
            
            # Create test configuration
            test_config = GoProConfiguration(
                video_resolution=VideoResolution.RES_2_7K,
                frame_rate=FrameRate.FPS_30,
                hindsight=HindsightOption.SEC_30,
                config_name="Integration Test Config"
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "upload_test_config.yaml"
                
                # Save configuration to file
                with open(config_path, 'w') as f:
                    yaml.dump(test_config.to_dict(), f)
                
                # Upload configuration
                result = ctrl.upload_configuration(str(config_path))
                
                assert result is True
                
            ctrl.disconnect()
    
    def test_download_upload_roundtrip(self):
        """Test downloading and then uploading the same configuration."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            # Initialize controller and connect
            ctrl = GoProController()
            ctrl.connect()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                download_path = Path(temp_dir) / "downloaded_config.yaml"
                
                # Download configuration
                saved_path = ctrl.download_configuration(str(download_path))
                assert Path(saved_path).exists()
                
                # Modify the downloaded configuration
                with open(download_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                config_data['config_name'] = "Modified Integration Test Config"
                config_data['video_resolution'] = "1080p"
                
                upload_path = Path(temp_dir) / "modified_config.yaml"
                with open(upload_path, 'w') as f:
                    yaml.dump(config_data, f)
                
                # Upload modified configuration
                result = ctrl.upload_configuration(str(upload_path))
                assert result is True
                
            ctrl.disconnect()
    
    def test_configuration_validation_integration(self):
        """Test configuration validation in integration context."""
        # Test validation without needing GoPro connection
        ctrl = GoProController()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test valid configuration
            valid_config = GoProConfiguration(
                config_name="Valid Integration Config",
                video_resolution=VideoResolution.RES_4K
            )
            
            valid_path = Path(temp_dir) / "valid_config.yaml"
            with open(valid_path, 'w') as f:
                yaml.dump(valid_config.to_dict(), f)
            
            is_valid, message = ctrl.validate_configuration(str(valid_path))
            assert is_valid is True
            assert "valid" in message.lower()
            
            # Test invalid configuration
            invalid_config = {
                "video_resolution": "Invalid Resolution Value",
                "frame_rate": "999999",
                "config_name": "Invalid Config"
            }
            
            invalid_path = Path(temp_dir) / "invalid_config.yaml"
            with open(invalid_path, 'w') as f:
                yaml.dump(invalid_config, f)
            
            is_valid, message = ctrl.validate_configuration(str(invalid_path))
            assert is_valid is False
            assert len(message) > 0
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            ctrl = GoProController()
            
            # Test download without connection
            try:
                ctrl.download_configuration()
                assert False, "Should have raised ConnectionError"
            except ConnectionError as e:
                assert "No GoPro connection available" in str(e)
            
            # Test upload without connection
            test_config = GoProConfiguration(config_name="Error Test Config")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(test_config.to_dict(), f)
                temp_path = f.name
            
            try:
                ctrl.upload_configuration(temp_path)
                assert False, "Should have raised ConnectionError"
            except ConnectionError as e:
                assert "No GoPro connection available" in str(e)
            finally:
                Path(temp_path).unlink(missing_ok=True)
    
    def test_file_permissions_integration(self):
        """Test file permission handling in integration context."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            ctrl = GoProController()
            ctrl.connect()
            
            # Test downloading to a path that would cause permission error
            with mock.patch('builtins.open', side_effect=PermissionError("Permission denied")):
                try:
                    ctrl.download_configuration("/readonly/test_config.yaml")
                    assert False, "Should have raised OSError"
                except OSError as e:
                    assert "Permission denied" in str(e) or "Unable to save" in str(e)
            
            ctrl.disconnect()
    
    def test_yaml_format_integration(self):
        """Test YAML format handling in integration context."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            ctrl = GoProController()
            ctrl.connect()
            
            # Test uploading invalid YAML
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("invalid: yaml: content: [unclosed bracket")
                temp_path = f.name
            
            try:
                ctrl.upload_configuration(temp_path)
                assert False, "Should have raised ValueError for invalid YAML"
            except ValueError as e:
                assert "Invalid YAML format" in str(e) or "YAML" in str(e)
            finally:
                Path(temp_path).unlink(missing_ok=True)
                
            ctrl.disconnect()
    
    def test_configuration_with_unicode_integration(self):
        """Test handling unicode characters in configuration names."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            ctrl = GoProController()
            ctrl.connect()
            
            # Test configuration with unicode characters
            unicode_config = GoProConfiguration(
                config_name="测试配置 🎥📹 Integration Test",
                video_resolution=VideoResolution.RES_4K
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "unicode_config.yaml"
                
                # Save unicode configuration
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(unicode_config.to_dict(), f, allow_unicode=True)
                
                # Upload unicode configuration
                result = ctrl.upload_configuration(str(config_path))
                assert result is True
                
                # Verify the file can be read back correctly
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_data = yaml.safe_load(f)
                
                assert loaded_data['config_name'] == unicode_config.config_name
                
            ctrl.disconnect()
    
    def test_default_path_generation(self):
        """Test default path generation for configuration downloads."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            ctrl = GoProController()
            ctrl.connect()
            
            # Download with default path (no path specified)
            saved_path = ctrl.download_configuration()
            
            # Verify file exists and has expected naming pattern
            assert Path(saved_path).exists()
            assert saved_path.startswith("gopro_config_")
            assert saved_path.endswith(".yaml")
            
            # Clean up
            Path(saved_path).unlink(missing_ok=True)
            ctrl.disconnect()
    
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Path handling different on Windows")
    def test_nested_directory_creation(self):
        """Test creating nested directories for configuration files."""
        with mock.patch('GoProController.WiredGoPro', FakeGoPro):
            ctrl = GoProController()
            ctrl.connect()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                nested_path = Path(temp_dir) / "configs" / "gopro" / "test_nested.yaml"
                
                # Download to nested path (should create directories)
                saved_path = ctrl.download_configuration(str(nested_path))
                
                assert Path(saved_path).exists()
                assert saved_path == str(nested_path)
                assert nested_path.parent.exists()
                
            ctrl.disconnect()


class TestGoProConfigurationErrorScenarios:
    """Integration tests for error scenarios and edge cases."""
    
    def test_corrupted_file_recovery(self):
        """Test handling of corrupted configuration files."""
        ctrl = GoProController()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write partially corrupted YAML
            f.write("config_name: Test Config\nvideo_resolution: 4K\nframe_rate")
            temp_path = f.name
        
        try:
            is_valid, message = ctrl.validate_configuration(temp_path)
            assert is_valid is False
            assert len(message) > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_empty_configuration_file(self):
        """Test handling of empty configuration files."""
        ctrl = GoProController()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write empty file
            temp_path = f.name
        
        try:
            is_valid, message = ctrl.validate_configuration(temp_path)
            assert is_valid is False
            assert "empty" in message.lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_large_configuration_file(self):
        """Test handling of unusually large configuration files."""
        ctrl = GoProController()
        
        # Create configuration with large description
        large_config = GoProConfiguration(
            config_name="Large Config Test",
            video_resolution=VideoResolution.RES_4K
        )
        config_dict = large_config.to_dict()
        
        # Add large amount of data (simulating corrupted file)
        config_dict['large_field'] = 'A' * 10000  # 10KB of 'A' characters
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # This should fail validation due to extra field
            is_valid, message = ctrl.validate_configuration(temp_path)
            assert is_valid is False
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])