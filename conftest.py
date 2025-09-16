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
    parser.addoption(
        "--run-physical-gopro",
        action="store_true",
        default=False,
        help="Run tests that require physical GoPro hardware (GoPro cameras)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "physical_ble: tests that require physical BLE hardware - use --run-physical-ble to enable"
    )
    config.addinivalue_line(
        "markers",
        "physical_gopro: tests that require physical GoPro hardware - use --run-physical-gopro to enable"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip physical hardware tests unless explicitly requested.

    This automatically skips any test marked with @pytest.mark.physical_ble or
    @pytest.mark.physical_gopro unless the corresponding flag is provided.
    """
    # Skip physical BLE tests unless flag is provided
    if not config.getoption("--run-physical-ble"):
        skip_ble = pytest.mark.skip(
            reason="Physical BLE hardware test - use --run-physical-ble flag to enable"
        )
        for item in items:
            if "physical_ble" in item.keywords:
                item.add_marker(skip_ble)

    # Skip physical GoPro tests unless flag is provided
    if not config.getoption("--run-physical-gopro"):
        skip_gopro = pytest.mark.skip(
            reason="Physical GoPro hardware test - use --run-physical-gopro flag to enable"
        )
        for item in items:
            if "physical_gopro" in item.keywords:
                item.add_marker(skip_gopro)