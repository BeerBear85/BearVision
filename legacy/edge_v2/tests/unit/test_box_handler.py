"""
Unit tests for BoxHandler using mocked Box SDK.

This test file focuses on testing the core upload/download/delete functionality
of BoxHandler using mocks. It does NOT make actual network calls and
runs quickly as part of the standard test suite.

For actual Box API integration testing, see test_box_drive.py
For authentication-specific testing, see test_box_authenticate.py
"""

import sys
from pathlib import Path
from unittest import mock
import pytest

from tests.stubs.box_sdk import FakeBoxClient, install_box_stubs

install_box_stubs()

MODULE_DIR = Path(__file__).resolve().parents[2] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))
from BoxHandler import BoxHandler


@mock.patch.object(BoxHandler, '_authenticate', lambda self: setattr(self, 'client', FakeBoxClient()))
def test_upload_download_delete(tmp_path):
    handler = BoxHandler({'BOX': {'root_folder': 'root'}})
    handler.connect()

    src = tmp_path / 'src.txt'
    src.write_text('hello')

    handler.upload_file(str(src), 'folder/file.txt')

    dst = tmp_path / 'dst.txt'
    handler.download_file('folder/file.txt', str(dst))
    assert dst.read_text() == 'hello'

    handler.delete_file('folder/file.txt')
    with pytest.raises(FileNotFoundError):
        handler.download_file('folder/file.txt', str(tmp_path / 'missing.txt'))


@mock.patch.object(BoxHandler, '_authenticate', lambda self: setattr(self, 'client', FakeBoxClient()))
def test_list_files(tmp_path):
    handler = BoxHandler({'BOX': {'root_folder': 'root'}})
    handler.connect()

    (tmp_path / 'src1.txt').write_text('a')
    (tmp_path / 'src2.txt').write_text('b')

    handler.upload_file(str(tmp_path / 'src1.txt'), 'folder/file1.txt')
    handler.upload_file(str(tmp_path / 'src2.txt'), 'folder/file2.txt')

    files = handler.list_files('folder')
    assert sorted(files) == ['file1.txt', 'file2.txt']
