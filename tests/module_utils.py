from pathlib import Path
import importlib
import sys
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))


def import_module_safe(name: str):
    """Import module and skip tests if optional dependencies are missing."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Skipping {name} due to missing dependency: {exc.name}")
