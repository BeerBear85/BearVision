"""
GoPro Configuration Model

Pydantic-based configuration model for GoPro camera settings.
Supports validation and serialization to/from YAML files.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class VideoResolution(str, Enum):
    """Video resolution options."""
    RES_4K = "4K"
    RES_2_7K = "2.7K"
    RES_1080P = "1080p"
    RES_5_3K = "5.3K"


class FrameRate(str, Enum):
    """Frame rate options."""
    FPS_24 = "24"
    FPS_25 = "25" 
    FPS_30 = "30"
    FPS_50 = "50"
    FPS_60 = "60"
    FPS_100 = "100"
    FPS_120 = "120"
    FPS_200 = "200"
    FPS_240 = "240"


class VideoLens(str, Enum):
    """Video lens/field of view options."""
    WIDE = "Wide"
    LINEAR = "Linear"
    NARROW = "Narrow"
    SUPERVIEW = "SuperView"
    HYPERVIEW = "HyperView"
    MAX_SUPERVIEW = "Max SuperView"


class HindsightOption(str, Enum):
    """Hindsight buffer options."""
    OFF = "OFF"
    SEC_15 = "15 seconds"
    SEC_30 = "30 seconds"


class HyperSmooth(str, Enum):
    """HyperSmooth stabilization options."""
    OFF = "OFF"
    LOW = "Low"
    STANDARD = "Standard"
    HIGH = "High"
    BOOST = "Boost"
    AUTO_BOOST = "Auto Boost"


class GPS(str, Enum):
    """GPS options."""
    ON = "ON"
    OFF = "OFF"


class VideoBitRate(str, Enum):
    """Video bit rate options."""
    STANDARD = "Standard"
    HIGH = "High"


class VideoPerformanceMode(str, Enum):
    """Video performance mode options."""
    EXTENDED_BATTERY = "Extended Battery"
    MAXIMUM_VIDEO = "Maximum Video Performance"
    TRIPOD_STATIONARY = "Tripod Stationary Video"


class GoProConfiguration(BaseModel):
    """
    GoPro camera configuration model.
    
    This model represents the configurable settings for a GoPro camera
    and provides validation for configuration values.
    """
    
    # Video settings
    video_resolution: Optional[VideoResolution] = Field(
        default=VideoResolution.RES_4K,
        description="Video recording resolution"
    )
    
    frame_rate: Optional[FrameRate] = Field(
        default=FrameRate.FPS_60,
        description="Video recording frame rate"
    )
    
    video_lens: Optional[VideoLens] = Field(
        default=VideoLens.WIDE,
        description="Video lens/field of view setting"
    )
    
    hindsight: Optional[HindsightOption] = Field(
        default=HindsightOption.SEC_15,
        description="HindSight buffer duration"
    )
    
    # Image stabilization
    hypersmooth: Optional[HyperSmooth] = Field(
        default=HyperSmooth.HIGH,
        description="HyperSmooth image stabilization level"
    )
    
    # Location and connectivity
    gps: Optional[GPS] = Field(
        default=GPS.ON,
        description="GPS tracking enable/disable"
    )
    
    # Performance settings
    video_bit_rate: Optional[VideoBitRate] = Field(
        default=VideoBitRate.HIGH,
        description="Video encoding bit rate"
    )
    
    video_performance_mode: Optional[VideoPerformanceMode] = Field(
        default=VideoPerformanceMode.MAXIMUM_VIDEO,
        description="Video performance optimization mode"
    )
    
    # Metadata
    config_name: Optional[str] = Field(
        default="Default Configuration",
        description="Human-readable name for this configuration"
    )
    
    config_version: str = Field(
        default="1.0",
        description="Configuration schema version"
    )
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"  # Don't allow extra fields
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        # Use mode='json' to ensure enum values are serialized as their values, not objects
        return {k: v for k, v in self.model_dump(mode='json').items() if v is not None}
    
    @classmethod
    def from_gopro_state(cls, camera_state: dict) -> "GoProConfiguration":
        """
        Create configuration from GoPro camera state.
        
        Args:
            camera_state: Dictionary containing camera state from get_camera_state()
            
        Returns:
            GoProConfiguration instance
        """
        # Map GoPro state values to our configuration model
        # This would need to be implemented based on actual GoPro state format
        settings = camera_state.get("settings", {})
        
        config_data = {}
        
        # Map video resolution (example mapping)
        if "video_resolution" in settings:
            # Convert from GoPro internal values to our enum values
            res_mapping = {
                12: VideoResolution.RES_4K,
                9: VideoResolution.RES_2_7K,
                6: VideoResolution.RES_1080P,
                # Add more mappings as needed
            }
            config_data["video_resolution"] = res_mapping.get(settings["video_resolution"])
        
        # Map frame rate (example mapping)  
        if "frames_per_second" in settings:
            fps_mapping = {
                0: FrameRate.FPS_24,
                1: FrameRate.FPS_25,
                2: FrameRate.FPS_30,
                8: FrameRate.FPS_60,
                # Add more mappings as needed
            }
            config_data["frame_rate"] = fps_mapping.get(settings["frames_per_second"])
        
        # Add more field mappings as needed
        
        return cls(**config_data)
    
    def to_gopro_settings(self) -> dict:
        """
        Convert configuration to GoPro HTTP settings format.
        
        Returns:
            Dictionary of setting names and values for GoPro HTTP API
        """
        settings = {}
        
        if self.video_resolution:
            # Map our enum values to GoPro internal values
            res_mapping = {
                VideoResolution.RES_4K: 12,
                VideoResolution.RES_2_7K: 9,
                VideoResolution.RES_1080P: 6,
                # Add more mappings as needed
            }
            settings["video_resolution"] = res_mapping.get(self.video_resolution)
        
        if self.frame_rate:
            fps_mapping = {
                FrameRate.FPS_24: 0,
                FrameRate.FPS_25: 1,
                FrameRate.FPS_30: 2,
                FrameRate.FPS_60: 8,
                # Add more mappings as needed
            }
            settings["frames_per_second"] = fps_mapping.get(self.frame_rate)
        
        # Add more mappings as needed
        
        return settings


def load_config_from_yaml(file_path: str) -> GoProConfiguration:
    """
    Load configuration from YAML file.
    
    Args:
        file_path: Path to YAML configuration file
        
    Returns:
        Validated GoProConfiguration instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or contains invalid YAML
        ValidationError: If configuration is invalid
    """
    import yaml
    from pathlib import Path
    from pydantic import ValidationError
    
    config_file = Path(file_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {file_path}: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Unable to read file {file_path} (encoding issue): {e}")
    except Exception as e:
        raise ValueError(f"Error reading configuration file {file_path}: {e}")
    
    if config_data is None:
        raise ValueError(f"Configuration file {file_path} is empty")
    
    if not isinstance(config_data, dict):
        raise ValueError(f"Configuration file {file_path} must contain a dictionary/object, got {type(config_data).__name__}")
    
    try:
        return GoProConfiguration(**config_data)
    except ValidationError as e:
        # Re-raise the validation error with original details
        # The error message can be processed by the caller if needed
        raise


def save_config_to_yaml(config: GoProConfiguration, file_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: GoProConfiguration instance to save
        file_path: Path to save YAML file
        
    Raises:
        PermissionError: If unable to write to file path
        OSError: If unable to create directories or write file
        ValueError: If config data cannot be serialized to YAML
    """
    import yaml
    from pathlib import Path
    
    if not isinstance(config, GoProConfiguration):
        raise ValueError(f"Expected GoProConfiguration instance, got {type(config).__name__}")
    
    config_file = Path(file_path)
    
    try:
        # Create parent directories if they don't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise OSError(f"Unable to create directory {config_file.parent}: {e}")
    
    try:
        config_dict = config.to_dict()
    except Exception as e:
        raise ValueError(f"Unable to convert configuration to dictionary: {e}")
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing to {file_path}: {e}")
    except OSError as e:
        raise OSError(f"Error writing configuration file {file_path}: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error serializing configuration to YAML: {e}")
    except Exception as e:
        raise OSError(f"Unexpected error writing configuration file {file_path}: {e}")