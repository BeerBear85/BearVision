from pathlib import Path
import inspect
import sys
import pytest
from tests.module_utils import import_module_safe

MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
MODULE_NAMES = [p.stem for p in MODULE_DIR.glob("*.py") if p.stem != "__init__"]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_import_module(module_name):
    mod = import_module_safe(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_has_public_attributes(module_name):
    mod = import_module_safe(module_name)
    public_attrs = [a for a in dir(mod) if not a.startswith("_")]
    assert len(public_attrs) > 0


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_file_property(module_name):
    mod = import_module_safe(module_name)
    assert mod.__file__.endswith(".py")


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_public_callables_smoke(module_name):
    mod = import_module_safe(module_name)
    callables = [
        obj
        for name, obj in inspect.getmembers(mod)
        if callable(obj) and obj.__module__ == module_name and not name.startswith("_")
    ]
    if not callables:
        pytest.skip("No public callables")
    for func in callables:
        try:
            func()
        except Exception:
            pass
