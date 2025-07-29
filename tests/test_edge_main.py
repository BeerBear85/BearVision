import sys
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / 'code' / 'Application'
sys.path.append(str(APP_DIR))

import edge_main


def test_main_starts_threads():
    before = set(threading.enumerate())
    threads = edge_main.main()
    after = set(threading.enumerate())
    # Ensure new threads started
    assert len(after - before) >= 2
    for t in threads:
        assert t.is_alive()
