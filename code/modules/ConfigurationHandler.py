import os
import logging
from configparser import ConfigParser
from shutil import copyfile


logger = logging.getLogger(__name__)  # Set logger to reflect the current file

_last_saved_config_filename = "last_used_config.ini"  # Should be the only parameter in code

_configuration = None
_configuration_path = None


class ConfigurationHandler:
    def __init__(self):
        self.parser = ConfigParser()
        return

    @staticmethod
    def read_last_used_config_file():
        global _configuration
        global _configuration_path
        if os.path.isfile(_last_saved_config_filename):
            ConfigurationHandler.read_config_file(_last_saved_config_filename)
            _configuration_path = os.path.dirname(_last_saved_config_filename)
            return True
        return False

    @staticmethod
    def read_config_file(arg_config_file_path):
        global _configuration
        if _configuration is None:
            _configuration = ConfigParser()

        if not os.path.isfile(arg_config_file_path):
            logger.warning("Conf (%s) not found. Using defaults." % arg_config_file_path)

        _configuration.read(arg_config_file_path)

        # Save new config
        if not os.path.isfile(_last_saved_config_filename) or (os.path.isfile(_last_saved_config_filename) and (not os.path.samefile(arg_config_file_path, _last_saved_config_filename))):
            copyfile(arg_config_file_path, _last_saved_config_filename)
        return _configuration

    @staticmethod
    def get_configuration():
        global _configuration
        if _configuration is None:
            print("No configuration was found!")
        return _configuration

    @staticmethod
    def get_configuration_path():
        global _configuration_path
        if _configuration_path is None:
            print("No configuration was found!")
        return _configuration_path
