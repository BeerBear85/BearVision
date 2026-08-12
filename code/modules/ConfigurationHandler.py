import os
import logging
from configparser import ConfigParser
from shutil import copyfile
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)  # Set logger to reflect the current file

_last_saved_config_filename = "last_used_config.ini"  # Should be the only parameter in code

_configuration = None
_configuration_path = None


class ConfigurationHandler:
    """Utility functions for reading and caching configuration files."""

    def __init__(self):
        """Create a new handler instance.

        The class mostly exposes static methods, but a parser instance is kept
        for potential future extensions.

        Returns:
            None
        """
        self.parser = ConfigParser()
        return

    @staticmethod
    def read_last_used_config_file():
        """Load the configuration that was used in the previous run.

        The last configuration file path is stored in
        ``_last_saved_config_filename``. Keeping this information allows the
        application to start with sensible defaults without user interaction.

        Returns:
            bool: ``True`` if a configuration was loaded, ``False`` otherwise.
        """
        global _configuration
        global _configuration_path
        if os.path.isfile(_last_saved_config_filename):
            ConfigurationHandler.read_config_file(_last_saved_config_filename)
            _configuration_path = os.path.join(os.getcwd(), _last_saved_config_filename)
            return True
        return False

    @staticmethod
    def read_config_file(arg_config_file_path):
        """Parse a configuration file and cache its contents.

        Args:
            arg_config_file_path (str): Path to the configuration file.

        Returns:
            ConfigParser: The parsed configuration object.
        """
        global _configuration
        global _configuration_path
        if not os.path.isfile(arg_config_file_path):
            # Falling back to defaults is preferable to failing hard here so
            # that the application can still start in a limited form.
            logger.warning("Conf (%s) not found. Using defaults." % arg_config_file_path)
            _configuration = ConfigParser()
            _configuration_path = arg_config_file_path
            return _configuration

        _configuration_path = arg_config_file_path
        _configuration = ConfigParser()

        source = Path(arg_config_file_path)
        text = source.read_text(encoding="utf-8")
        is_versioned_yaml = (
            source.suffix.lower() in {".yaml", ".yml"}
            or text.lstrip().startswith("config_schema_version:")
        )
        if is_versioned_yaml:
            data = yaml.safe_load(text) or {}
            if data.get("config_schema_version") != "2.0":
                raise ValueError("unsupported or missing configuration schema version")
            if data.get("config_kind") != "bearvision-edge":
                raise ValueError("ConfigurationHandler only accepts bearvision-edge configuration")
            _configuration.read_dict(ConfigurationHandler._legacy_sections(data))
        else:
            _configuration.read_string(text)

        # Store the loaded configuration path to disk so that future runs can
        # reuse it automatically.
        if not os.path.isfile(_last_saved_config_filename) or (
            os.path.isfile(_last_saved_config_filename)
            and (not os.path.samefile(arg_config_file_path, _last_saved_config_filename))
        ):
            copyfile(arg_config_file_path, _last_saved_config_filename)
        return _configuration

    @staticmethod
    def _legacy_sections(data):
        """Translate the versioned edge schema for unchanged legacy modules."""
        recording = data.get("recording", {})
        detection = data.get("detection", {})
        performance = data.get("performance", {})
        recovery = data.get("error_recovery", {})
        features = data.get("features", {})
        storage = data.get("storage", {})
        return {
            "EDGE_APPLICATION": {
                "post_detection_duration": recording.get("post_detection_duration_s", 5.0),
                "hindsight_mode_enabled": recording.get("hindsight_enabled", True),
                "yolo_enabled": detection.get("enabled", True),
                "yolo_model": detection.get("model", "yolov8n"),
                "detection_confidence_threshold": detection.get("confidence_threshold", 0.5),
                "detection_cooldown": detection.get("cooldown_s", 2.0),
                "stream_max_fps": performance.get("max_fps", 30),
                "stream_max_lag_ms": performance.get("max_lag_ms", 250),
                "stream_buffer_drain": performance.get("buffer_drain", True),
                "stream_callback_queue_size": performance.get("callback_queue_size", 5),
                "max_error_restarts": recovery.get("max_restarts", 5),
                "error_restart_delay": recovery.get("restart_delay_s", 2.0),
                "enable_ble_logging": features.get("ble_logging", True),
                "enable_post_processing": features.get("post_processing", True),
                "enable_cloud_upload": features.get("cloud_upload", True),
                "preview_stream_enabled": features.get("preview_stream", True),
            },
            "STORAGE_COMMON": {
                "secret_key_name": storage.get("credential_env", "STORAGE_CREDENTIALS_B64"),
                "secret_key_name_2": storage.get("secondary_credential_env", ""),
            },
            "BOX": {"root_folder": storage.get("root_folder", "bearvision_files")},
        }

    @staticmethod
    def get_configuration():
        """Return the currently cached configuration.

        Returns:
            ConfigParser: The cached configuration instance or ``None`` if no
            configuration has been loaded yet.
        """
        global _configuration
        if _configuration is None:
            print("No configuration was found!")
        return _configuration

    @staticmethod
    def get_configuration_path():
        """Return the directory containing the active configuration file.

        Returns:
            str: Directory path of the loaded configuration or ``None`` if no
            configuration has been loaded yet.
        """
        global _configuration_path
        if _configuration_path is None:
            print("No configuration was found!")
        return os.path.dirname(_configuration_path)
