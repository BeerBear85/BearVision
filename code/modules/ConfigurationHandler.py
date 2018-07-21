import os
import logging
from configparser import ConfigParser


logger = logging.getLogger(__name__)  # Set logger to reflect the current file

_configuration = None

class ConfigurationHandler:
    def __init__(self):
        self.parser = ConfigParser()
        return

    @staticmethod
    def read_config_file(arg_config_file_path):
        global _configuration
        if _configuration == None:
            _configuration = ConfigParser()

        if not os.path.isfile(arg_config_file_path):
            logger.warning("Conf (%s) not found. Using defaults." % arg_config_file_path)

        _configuration.read(arg_config_file_path)
        return _configuration

    @staticmethod
    def get_configuration():
        global _configuration
        return _configuration