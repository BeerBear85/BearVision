import sys
import threading
from pathlib import Path
from unittest import mock

APP_DIR = Path(__file__).resolve().parents[1] / 'code' / 'Application'
sys.path.append(str(APP_DIR))

import edge_main


def test_main_starts_threads_and_sets_up():
    with mock.patch.object(edge_main.ConfigurationHandler, 'read_config_file') as read_cfg, \
         mock.patch.object(edge_main, 'GoProController') as GC:
        ctrl = GC.return_value

        before = set(threading.enumerate())
        threads = edge_main.main()
        after = set(threading.enumerate())

        cfg_path = Path(__file__).resolve().parents[1] / 'config.ini'
        read_cfg.assert_called_once_with(str(cfg_path))
        ctrl.connect.assert_called_once()
        ctrl.configure.assert_called_once()
        ctrl.start_preview.assert_called_once()

        assert len(after - before) >= 2
        for t in threads:
            assert t.is_alive()
