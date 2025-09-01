"""
pytest configuration file for BearVision project.

Defines custom command-line options and test collection behavior.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--run-physical-ble",
        action="store_true",
        default=False,
        help="Run tests that require physical BLE hardware (KBPro tags)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "physical_ble: tests that require physical BLE hardware - use --run-physical-ble to enable"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip physical_ble tests unless explicitly requested.
    
    This automatically skips any test marked with @pytest.mark.physical_ble
    unless the --run-physical-ble flag is provided on the command line.
    """
    if not config.getoption("--run-physical-ble"):
        skip_ble = pytest.mark.skip(
            reason="Physical BLE hardware test - use --run-physical-ble flag to enable"
        )
        
        for item in items:
            if "physical_ble" in item.keywords:
                item.add_marker(skip_ble)