"""Helper package for BearVision modules."""
import importlib
import sys

_cfg = importlib.import_module('ConfigurationHandler')
sys.modules.setdefault(__name__ + '.ConfigurationHandler', _cfg)
