"""
Edge Application Configuration Module

Provides centralized configuration management for the Edge Application system.
Loads parameters from INI file and provides typed access to configuration values.
"""

import logging
from typing import Optional
from configparser import ConfigParser
from pathlib import Path


logger = logging.getLogger(__name__)


class EdgeApplicationConfig:
    """
    Configuration manager for Edge Application.

    Loads and validates configuration parameters from INI file,
    providing typed access and sensible defaults.
    """

    # Configuration section name
    SECTION_NAME = "EDGE_APPLICATION"

    # Default configuration values
    DEFAULTS = {
        # YOLO Detection Settings
        "yolo_enabled": True,
        "yolo_model": "yolov8n",

        # Recording Settings
        "recording_duration": 30.0,

        # Error Recovery Settings
        "max_error_restarts": 3,
        "error_restart_delay": 2.0,

        # Thread Settings
        "enable_ble_logging": True,
        "enable_post_processing": True,
        "enable_cloud_upload": True,

        # State Machine Behavior
        "hindsight_mode_enabled": True,
        "preview_stream_enabled": True,

        # Detection Settings
        "detection_confidence_threshold": 0.5,
        "detection_cooldown": 2.0,
    }

    # Valid YOLO model names
    VALID_YOLO_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

    def __init__(self):
        """Initialize configuration with defaults."""
        self.config: Optional[ConfigParser] = None
        self.config_path: Optional[Path] = None
        self._values = self.DEFAULTS.copy()

    def load_from_file(self, config_path: str) -> bool:
        """
        Load configuration from INI file.

        Parameters
        ----------
        config_path : str
            Path to configuration INI file

        Returns
        -------
        bool
            True if configuration loaded successfully, False otherwise
        """
        try:
            self.config_path = Path(config_path)

            if not self.config_path.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return False

            self.config = ConfigParser()
            self.config.read(config_path)

            if not self.config.has_section(self.SECTION_NAME):
                logger.warning(f"No [{self.SECTION_NAME}] section found, using defaults")
                return False

            # Load configuration values
            self._load_values()

            # Validate configuration
            if not self.validate():
                logger.error("Configuration validation failed")
                return False

            logger.info(f"Configuration loaded from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def _load_values(self):
        """Load values from ConfigParser into typed dictionary."""
        if not self.config or not self.config.has_section(self.SECTION_NAME):
            return

        section = self.config[self.SECTION_NAME]

        # YOLO Detection Settings
        self._values["yolo_enabled"] = section.getboolean("yolo_enabled",
                                                          fallback=self.DEFAULTS["yolo_enabled"])
        self._values["yolo_model"] = section.get("yolo_model",
                                                 fallback=self.DEFAULTS["yolo_model"])

        # Recording Settings
        self._values["recording_duration"] = section.getfloat("recording_duration",
                                                              fallback=self.DEFAULTS["recording_duration"])

        # Error Recovery Settings
        self._values["max_error_restarts"] = section.getint("max_error_restarts",
                                                            fallback=self.DEFAULTS["max_error_restarts"])
        self._values["error_restart_delay"] = section.getfloat("error_restart_delay",
                                                               fallback=self.DEFAULTS["error_restart_delay"])

        # Thread Settings
        self._values["enable_ble_logging"] = section.getboolean("enable_ble_logging",
                                                                fallback=self.DEFAULTS["enable_ble_logging"])
        self._values["enable_post_processing"] = section.getboolean("enable_post_processing",
                                                                    fallback=self.DEFAULTS["enable_post_processing"])
        self._values["enable_cloud_upload"] = section.getboolean("enable_cloud_upload",
                                                                 fallback=self.DEFAULTS["enable_cloud_upload"])

        # State Machine Behavior
        self._values["hindsight_mode_enabled"] = section.getboolean("hindsight_mode_enabled",
                                                                    fallback=self.DEFAULTS["hindsight_mode_enabled"])
        self._values["preview_stream_enabled"] = section.getboolean("preview_stream_enabled",
                                                                    fallback=self.DEFAULTS["preview_stream_enabled"])

        # Detection Settings
        self._values["detection_confidence_threshold"] = section.getfloat("detection_confidence_threshold",
                                                                          fallback=self.DEFAULTS["detection_confidence_threshold"])
        self._values["detection_cooldown"] = section.getfloat("detection_cooldown",
                                                              fallback=self.DEFAULTS["detection_cooldown"])

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns
        -------
        bool
            True if all configuration values are valid
        """
        try:
            # Validate YOLO model
            if self._values["yolo_model"] not in self.VALID_YOLO_MODELS:
                logger.error(f"Invalid YOLO model: {self._values['yolo_model']}, "
                           f"must be one of {self.VALID_YOLO_MODELS}")
                return False

            # Validate recording duration
            if self._values["recording_duration"] <= 0:
                logger.error(f"Invalid recording_duration: {self._values['recording_duration']}, must be > 0")
                return False

            # Validate max error restarts
            if self._values["max_error_restarts"] < 0:
                logger.error(f"Invalid max_error_restarts: {self._values['max_error_restarts']}, must be >= 0")
                return False

            # Validate error restart delay
            if self._values["error_restart_delay"] < 0:
                logger.error(f"Invalid error_restart_delay: {self._values['error_restart_delay']}, must be >= 0")
                return False

            # Validate detection confidence threshold
            if not 0.0 <= self._values["detection_confidence_threshold"] <= 1.0:
                logger.error(f"Invalid detection_confidence_threshold: {self._values['detection_confidence_threshold']}, "
                           f"must be between 0.0 and 1.0")
                return False

            # Validate detection cooldown
            if self._values["detection_cooldown"] < 0:
                logger.error(f"Invalid detection_cooldown: {self._values['detection_cooldown']}, must be >= 0")
                return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    # YOLO Detection Settings
    def get_yolo_enabled(self) -> bool:
        """Get whether YOLO detection is enabled."""
        return self._values["yolo_enabled"]

    def get_yolo_model(self) -> str:
        """Get YOLO model name."""
        return self._values["yolo_model"]

    # Recording Settings
    def get_recording_duration(self) -> float:
        """Get recording duration in seconds."""
        return self._values["recording_duration"]

    # Error Recovery Settings
    def get_max_error_restarts(self) -> int:
        """Get maximum number of error restart attempts."""
        return self._values["max_error_restarts"]

    def get_error_restart_delay(self) -> float:
        """Get delay in seconds before error restart."""
        return self._values["error_restart_delay"]

    # Thread Settings
    def get_enable_ble_logging(self) -> bool:
        """Get whether BLE logging is enabled."""
        return self._values["enable_ble_logging"]

    def get_enable_post_processing(self) -> bool:
        """Get whether post-processing is enabled."""
        return self._values["enable_post_processing"]

    def get_enable_cloud_upload(self) -> bool:
        """Get whether cloud upload is enabled."""
        return self._values["enable_cloud_upload"]

    # State Machine Behavior
    def get_hindsight_mode_enabled(self) -> bool:
        """Get whether hindsight mode is enabled."""
        return self._values["hindsight_mode_enabled"]

    def get_preview_stream_enabled(self) -> bool:
        """Get whether preview stream is enabled."""
        return self._values["preview_stream_enabled"]

    # Detection Settings
    def get_detection_confidence_threshold(self) -> float:
        """Get detection confidence threshold (0.0-1.0)."""
        return self._values["detection_confidence_threshold"]

    def get_detection_cooldown(self) -> float:
        """Get detection cooldown in seconds."""
        return self._values["detection_cooldown"]

    def get_all_values(self) -> dict:
        """
        Get all configuration values as dictionary.

        Returns
        -------
        dict
            Dictionary of all configuration values
        """
        return self._values.copy()

    def print_config(self):
        """Print current configuration to log."""
        logger.info("=" * 60)
        logger.info("Edge Application Configuration")
        logger.info("=" * 60)
        logger.info("YOLO Detection Settings:")
        logger.info(f"  yolo_enabled: {self._values['yolo_enabled']}")
        logger.info(f"  yolo_model: {self._values['yolo_model']}")
        logger.info("")
        logger.info("Recording Settings:")
        logger.info(f"  recording_duration: {self._values['recording_duration']} seconds")
        logger.info("")
        logger.info("Error Recovery Settings:")
        logger.info(f"  max_error_restarts: {self._values['max_error_restarts']}")
        logger.info(f"  error_restart_delay: {self._values['error_restart_delay']} seconds")
        logger.info("")
        logger.info("Thread Settings:")
        logger.info(f"  enable_ble_logging: {self._values['enable_ble_logging']}")
        logger.info(f"  enable_post_processing: {self._values['enable_post_processing']}")
        logger.info(f"  enable_cloud_upload: {self._values['enable_cloud_upload']}")
        logger.info("")
        logger.info("State Machine Behavior:")
        logger.info(f"  hindsight_mode_enabled: {self._values['hindsight_mode_enabled']}")
        logger.info(f"  preview_stream_enabled: {self._values['preview_stream_enabled']}")
        logger.info("")
        logger.info("Detection Settings:")
        logger.info(f"  detection_confidence_threshold: {self._values['detection_confidence_threshold']}")
        logger.info(f"  detection_cooldown: {self._values['detection_cooldown']} seconds")
        logger.info("=" * 60)
