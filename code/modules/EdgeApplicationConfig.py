"""
Edge Application Configuration Module

Provides centralized configuration management for the Edge Application system.
Loads parameters from INI or YAML files and provides typed access to configuration values.
"""

import logging
from typing import Optional
from configparser import ConfigParser
from pathlib import Path
import yaml


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
        "post_detection_duration": 30.0,  # Active recording duration AFTER detection (seconds)

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

        # Stream Performance Settings
        "stream_max_fps": 30,
        "stream_buffer_drain": True,
        "stream_max_lag_ms": 500,
        "stream_callback_queue_size": 2,
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
        Load configuration from INI or YAML file (auto-detects format).

        Parameters
        ----------
        config_path : str
            Path to configuration file (.ini or .yaml/.yml)

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

            # Auto-detect format based on file extension
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                return self.load_from_yaml(config_path)
            else:
                return self._load_from_ini(config_path)

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def _load_from_ini(self, config_path: str) -> bool:
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

            logger.info(f"Configuration loaded from INI: {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load INI configuration: {e}")
            return False

    def load_from_yaml(self, yaml_path: str) -> bool:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to configuration YAML file

        Returns
        -------
        bool
            True if configuration loaded successfully, False otherwise
        """
        try:
            yaml_file = Path(yaml_path)

            if not yaml_file.exists():
                logger.warning(f"YAML config file not found: {yaml_path}, using defaults")
                return False

            with open(yaml_file, 'r') as f:
                yaml_data = yaml.safe_load(f)

            if not yaml_data:
                logger.warning(f"Empty YAML file: {yaml_path}, using defaults")
                return False

            # Parse YAML structure into config values
            self._parse_yaml_to_values(yaml_data)

            # Validate configuration
            if not self.validate():
                logger.error("Configuration validation failed")
                return False

            logger.info(f"Configuration loaded from YAML: {yaml_path}")
            return True

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load YAML configuration: {e}")
            return False

    def _parse_yaml_to_values(self, yaml_data: dict) -> None:
        """
        Parse YAML data structure into internal values dictionary.

        Parameters
        ----------
        yaml_data : dict
            Parsed YAML data with nested structure
        """
        # Recording settings
        if 'recording' in yaml_data:
            rec = yaml_data['recording']
            # Support new name (post_detection_duration) with backward compatibility for old name (duration)
            if 'post_detection_duration' in rec:
                self._values['post_detection_duration'] = float(rec['post_detection_duration'])
            elif 'duration' in rec:
                # Backward compatibility
                self._values['post_detection_duration'] = float(rec['duration'])
                logger.warning("'duration' parameter is deprecated, use 'post_detection_duration' instead")
            if 'hindsight_enabled' in rec:
                self._values['hindsight_mode_enabled'] = bool(rec['hindsight_enabled'])

        # Detection settings
        if 'detection' in yaml_data:
            det = yaml_data['detection']
            if 'yolo_enabled' in det:
                self._values['yolo_enabled'] = bool(det['yolo_enabled'])
            if 'yolo_model' in det:
                self._values['yolo_model'] = str(det['yolo_model'])
            if 'confidence_threshold' in det:
                self._values['detection_confidence_threshold'] = float(det['confidence_threshold'])
            if 'cooldown' in det:
                self._values['detection_cooldown'] = float(det['cooldown'])

        # Performance settings
        if 'performance' in yaml_data:
            perf = yaml_data['performance']
            if 'stream_max_fps' in perf:
                self._values['stream_max_fps'] = int(perf['stream_max_fps'])
            if 'stream_max_lag_ms' in perf:
                self._values['stream_max_lag_ms'] = int(perf['stream_max_lag_ms'])

        # Error recovery settings
        if 'error_recovery' in yaml_data:
            err = yaml_data['error_recovery']
            if 'max_restarts' in err:
                self._values['max_error_restarts'] = int(err['max_restarts'])
            if 'restart_delay' in err:
                self._values['error_restart_delay'] = float(err['restart_delay'])

        # Thread settings
        if 'threads' in yaml_data:
            threads = yaml_data['threads']
            if 'enable_ble_logging' in threads:
                self._values['enable_ble_logging'] = bool(threads['enable_ble_logging'])
            if 'enable_cloud_upload' in threads:
                self._values['enable_cloud_upload'] = bool(threads['enable_cloud_upload'])

        # System settings
        if 'system' in yaml_data:
            sys = yaml_data['system']
            if 'log_level' in sys:
                # Store log level for potential future use
                log_level = str(sys['log_level']).upper()
                if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                    # Could set logging level here if needed
                    logger.info(f"Log level set to: {log_level}")

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
        # Support new name (post_detection_duration) with backward compatibility for old name (recording_duration)
        if section.get("post_detection_duration"):
            self._values["post_detection_duration"] = section.getfloat("post_detection_duration")
        elif section.get("recording_duration"):
            self._values["post_detection_duration"] = section.getfloat("recording_duration")
            logger.warning("'recording_duration' parameter is deprecated, use 'post_detection_duration' instead")
        else:
            self._values["post_detection_duration"] = self.DEFAULTS["post_detection_duration"]

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

        # Stream Performance Settings
        self._values["stream_max_fps"] = section.getint("stream_max_fps",
                                                        fallback=self.DEFAULTS["stream_max_fps"])
        self._values["stream_buffer_drain"] = section.getboolean("stream_buffer_drain",
                                                                 fallback=self.DEFAULTS["stream_buffer_drain"])
        self._values["stream_max_lag_ms"] = section.getint("stream_max_lag_ms",
                                                           fallback=self.DEFAULTS["stream_max_lag_ms"])
        self._values["stream_callback_queue_size"] = section.getint("stream_callback_queue_size",
                                                                    fallback=self.DEFAULTS["stream_callback_queue_size"])

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

            # Validate post-detection recording duration
            if self._values["post_detection_duration"] <= 0:
                logger.error(f"Invalid post_detection_duration: {self._values['post_detection_duration']}, must be > 0")
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

            # Validate stream max FPS
            if self._values["stream_max_fps"] <= 0 or self._values["stream_max_fps"] > 120:
                logger.error(f"Invalid stream_max_fps: {self._values['stream_max_fps']}, must be between 1 and 120")
                return False

            # Validate stream max lag
            if self._values["stream_max_lag_ms"] < 0:
                logger.error(f"Invalid stream_max_lag_ms: {self._values['stream_max_lag_ms']}, must be >= 0")
                return False

            # Validate stream callback queue size
            if self._values["stream_callback_queue_size"] < 1 or self._values["stream_callback_queue_size"] > 10:
                logger.error(f"Invalid stream_callback_queue_size: {self._values['stream_callback_queue_size']}, must be between 1 and 10")
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
    def get_post_detection_duration(self) -> float:
        """
        Get post-detection recording duration in seconds.

        This is the active recording time AFTER wakeboarder detection.
        Total clip length = 15s hindsight buffer + post_detection_duration
        """
        return self._values["post_detection_duration"]

    def get_recording_duration(self) -> float:
        """
        DEPRECATED: Use get_post_detection_duration() instead.
        Get post-detection recording duration in seconds.
        """
        logger.warning("get_recording_duration() is deprecated, use get_post_detection_duration() instead")
        return self.get_post_detection_duration()

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

    # Stream Performance Settings
    def get_stream_max_fps(self) -> int:
        """Get maximum stream processing FPS."""
        return self._values["stream_max_fps"]

    def get_stream_buffer_drain(self) -> bool:
        """Get whether stream buffer draining is enabled."""
        return self._values["stream_buffer_drain"]

    def get_stream_max_lag_ms(self) -> int:
        """Get maximum acceptable stream lag in milliseconds."""
        return self._values["stream_max_lag_ms"]

    def get_stream_callback_queue_size(self) -> int:
        """Get stream callback queue size."""
        return self._values["stream_callback_queue_size"]

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
        logger.info(f"  post_detection_duration: {self._values['post_detection_duration']} seconds (recording time AFTER detection)")
        logger.info(f"  Total clip length: ~{15 + self._values['post_detection_duration']} seconds (15s hindsight + {self._values['post_detection_duration']}s post-detection)")
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
        logger.info("")
        logger.info("Stream Performance Settings:")
        logger.info(f"  stream_max_fps: {self._values['stream_max_fps']}")
        logger.info(f"  stream_buffer_drain: {self._values['stream_buffer_drain']}")
        logger.info(f"  stream_max_lag_ms: {self._values['stream_max_lag_ms']} ms")
        logger.info(f"  stream_callback_queue_size: {self._values['stream_callback_queue_size']}")
        logger.info("=" * 60)
