"""
Unit tests for GoPro configuration handling.

Tests the GoProConfig module including:
- Pydantic model validation
- YAML file loading and saving
- Configuration serialization/deserialization
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest import mock
from pydantic import ValidationError

import sys
MODULE_DIR = Path(__file__).resolve().parents[2] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from GoProConfig import (
    GoProConfiguration, VideoResolution, FrameRate, VideoLens,
    HindsightOption, HyperSmooth, GPS, VideoBitRate, VideoPerformanceMode,
    load_config_from_yaml, save_config_to_yaml
)


class TestGoProConfiguration:
    """Test the GoProConfiguration Pydantic model."""
    
    def test_default_configuration(self):
        """Test creating configuration with default values."""
        config = GoProConfiguration()
        
        assert config.video_resolution == VideoResolution.RES_4K
        assert config.frame_rate == FrameRate.FPS_60
        assert config.video_lens == VideoLens.WIDE
        assert config.hindsight == HindsightOption.SEC_15
        assert config.hypersmooth == HyperSmooth.HIGH
        assert config.gps == GPS.ON
        assert config.video_bit_rate == VideoBitRate.HIGH
        assert config.video_performance_mode == VideoPerformanceMode.MAXIMUM_VIDEO
        assert config.config_name == "Default Configuration"
        assert config.config_version == "1.0"
    
    def test_custom_configuration(self):
        """Test creating configuration with custom values."""
        config = GoProConfiguration(
            video_resolution=VideoResolution.RES_1080P,
            frame_rate=FrameRate.FPS_30,
            video_lens=VideoLens.LINEAR,
            hindsight=HindsightOption.OFF,
            hypersmooth=HyperSmooth.LOW,
            gps=GPS.OFF,
            video_bit_rate=VideoBitRate.STANDARD,
            video_performance_mode=VideoPerformanceMode.EXTENDED_BATTERY,
            config_name="Custom Test Config",
            config_version="2.0"
        )
        
        assert config.video_resolution == VideoResolution.RES_1080P
        assert config.frame_rate == FrameRate.FPS_30
        assert config.video_lens == VideoLens.LINEAR
        assert config.hindsight == HindsightOption.OFF
        assert config.hypersmooth == HyperSmooth.LOW
        assert config.gps == GPS.OFF
        assert config.video_bit_rate == VideoBitRate.STANDARD
        assert config.video_performance_mode == VideoPerformanceMode.EXTENDED_BATTERY
        assert config.config_name == "Custom Test Config"
        assert config.config_version == "2.0"
    
    def test_invalid_values(self):
        """Test that invalid values raise ValidationError."""
        with pytest.raises(ValidationError):
            GoProConfiguration(video_resolution="Invalid Resolution")
        
        with pytest.raises(ValidationError):
            GoProConfiguration(frame_rate="999")
        
        with pytest.raises(ValidationError):
            GoProConfiguration(config_version=123)  # Should be string
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            GoProConfiguration(invalid_field="test")
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = GoProConfiguration(
            video_resolution=VideoResolution.RES_2_7K,
            frame_rate=FrameRate.FPS_24,
            config_name="Test Config"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["video_resolution"] == "2.7K"
        assert config_dict["frame_rate"] == "24"
        assert config_dict["config_name"] == "Test Config"
        assert "config_version" in config_dict
    
    def test_from_gopro_state(self):
        """Test creating configuration from GoPro camera state."""
        # Mock camera state data
        camera_state = {
            "settings": {
                "video_resolution": 12,  # 4K
                "frames_per_second": 8,  # 60 FPS
                "hindsight": 1  # 15 seconds
            },
            "status": {
                "encoding": False
            }
        }
        
        config = GoProConfiguration.from_gopro_state(camera_state)
        
        assert isinstance(config, GoProConfiguration)
        # Note: The actual mapping would depend on the implementation
        # This is a basic test of the interface
    
    def test_to_gopro_settings(self):
        """Test converting configuration to GoPro settings format."""
        config = GoProConfiguration(
            video_resolution=VideoResolution.RES_4K,
            frame_rate=FrameRate.FPS_60
        )
        
        settings = config.to_gopro_settings()
        
        assert isinstance(settings, dict)
        # Note: The actual values would depend on the implementation
        # This is a basic test of the interface


class TestYAMLOperations:
    """Test YAML file loading and saving operations."""
    
    def test_save_and_load_config(self):
        """Test saving configuration to YAML and loading it back."""
        config = GoProConfiguration(
            video_resolution=VideoResolution.RES_2_7K,
            frame_rate=FrameRate.FPS_30,
            video_lens=VideoLens.LINEAR,
            config_name="Test Save/Load Config"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            save_config_to_yaml(config, temp_path)
            
            # Verify file exists and contains YAML
            assert Path(temp_path).exists()
            with open(temp_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            assert isinstance(yaml_content, dict)
            assert yaml_content["video_resolution"] == "2.7K"
            assert yaml_content["frame_rate"] == "30"
            
            # Load configuration back
            loaded_config = load_config_from_yaml(temp_path)
            
            assert loaded_config.video_resolution == config.video_resolution
            assert loaded_config.frame_rate == config.frame_rate
            assert loaded_config.video_lens == config.video_lens
            assert loaded_config.config_name == config.config_name
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_nonexistent_file(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_yaml("nonexistent_file.yaml")
    
    def test_load_invalid_yaml(self):
        """Test loading configuration from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid YAML format"):
                load_config_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_empty_file(self):
        """Test loading configuration from empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write nothing (empty file)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="empty"):
                load_config_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_non_dict_yaml(self):
        """Test loading configuration from YAML that's not a dictionary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(["this", "is", "a", "list"], f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must contain a dictionary"):
                load_config_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_invalid_config_values(self):
        """Test loading configuration with invalid values."""
        invalid_config = {
            "video_resolution": "Invalid Resolution",
            "frame_rate": "999",
            "config_name": "Invalid Config"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError):
                load_config_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_save_invalid_config_type(self):
        """Test saving invalid configuration type."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Expected GoProConfiguration instance"):
                save_config_to_yaml("not a config object", temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_save_to_readonly_directory(self):
        """Test saving configuration to read-only directory."""
        # This test may not work on all systems, so we'll mock it
        config = GoProConfiguration()
        
        with mock.patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(OSError, match="Unable to create directory"):
                save_config_to_yaml(config, "/readonly/path/config.yaml")
    
    def test_save_with_unicode_characters(self):
        """Test saving configuration with unicode characters in config name."""
        config = GoProConfiguration(
            config_name="测试配置 🎥📹"  # Chinese characters and emojis
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save and load to ensure unicode is handled correctly
            save_config_to_yaml(config, temp_path)
            loaded_config = load_config_from_yaml(temp_path)
            
            assert loaded_config.config_name == config.config_name
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestEnumValues:
    """Test enum value constraints."""
    
    def test_video_resolution_enum(self):
        """Test VideoResolution enum values."""
        assert VideoResolution.RES_4K.value == "4K"
        assert VideoResolution.RES_2_7K.value == "2.7K"
        assert VideoResolution.RES_1080P.value == "1080p"
        assert VideoResolution.RES_5_3K.value == "5.3K"
    
    def test_frame_rate_enum(self):
        """Test FrameRate enum values."""
        assert FrameRate.FPS_24.value == "24"
        assert FrameRate.FPS_30.value == "30"
        assert FrameRate.FPS_60.value == "60"
        assert FrameRate.FPS_120.value == "120"
    
    def test_hindsight_enum(self):
        """Test HindsightOption enum values."""
        assert HindsightOption.OFF.value == "OFF"
        assert HindsightOption.SEC_15.value == "15 seconds"
        assert HindsightOption.SEC_30.value == "30 seconds"
    
    def test_gps_enum(self):
        """Test GPS enum values."""
        assert GPS.ON.value == "ON"
        assert GPS.OFF.value == "OFF"


if __name__ == "__main__":
    pytest.main([__file__])