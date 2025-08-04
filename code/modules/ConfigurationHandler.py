import os
import logging
from configparser import ConfigParser
from shutil import copyfile


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
        if _configuration is None:
            _configuration = ConfigParser()

        if not os.path.isfile(arg_config_file_path):
            # Falling back to defaults is preferable to failing hard here so
            # that the application can still start in a limited form.
            logger.warning("Conf (%s) not found. Using defaults." % arg_config_file_path)

        _configuration_path = arg_config_file_path
        _configuration.read(arg_config_file_path)

        # Store the loaded configuration path to disk so that future runs can
        # reuse it automatically.
        if not os.path.isfile(_last_saved_config_filename) or (
            os.path.isfile(_last_saved_config_filename)
            and (not os.path.samefile(arg_config_file_path, _last_saved_config_filename))
        ):
            copyfile(arg_config_file_path, _last_saved_config_filename)
        return _configuration

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
