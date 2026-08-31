import importlib
import inspect
import sys
import asyncio
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[2] / 'code' / 'modules'

# Ensure modules directory is on the path for imports
sys.path.append(str(MODULE_DIR))

def _call_func(func):
    # Skip functions that are known to block indefinitely or cause side effects
    blocking_functions = {
        'start_scan',  # BLE beacon scanning - infinite loop
        'init',        # DNN model loading - may hang on missing models
    }

    if func.__name__ in blocking_functions:
        return

    sig = inspect.signature(func)
    args = []
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            args.append(None)
    try:
        if inspect.iscoroutinefunction(func):
            # Skip async functions to avoid hanging during tests
            return
        func(*args)
    except BaseException:
        # Ignore any errors raised when calling with dummy args
        pass

def test_call_all_functions_and_methods():
    for mod_path in MODULE_DIR.glob('*.py'):
        if mod_path.name == '__init__.py':
            continue
        try:
            module = importlib.import_module(mod_path.stem)
        except BaseException:
            # Skip modules that fail to import due to missing dependencies
            continue
        for _, func in inspect.getmembers(module, inspect.isfunction):
            if func.__module__ != module.__name__:
                continue
            _call_func(func)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            try:
                init_sig = inspect.signature(cls)
                init_args = [None for p in init_sig.parameters.values() if p.default is inspect.Parameter.empty]
                instance = cls(*init_args)
            except BaseException:
                continue
            for _, method in inspect.getmembers(instance, inspect.ismethod):
                if method.__name__.startswith('_'):
                    continue
                _call_func(method)
