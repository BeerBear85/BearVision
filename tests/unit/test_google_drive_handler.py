"""
Unit tests for GoogleDriveHandler using mocked Google Drive API.

This test file focuses on testing the core upload/download/delete functionality
of GoogleDriveHandler using mocks. It does NOT make actual network calls and
runs quickly as part of the standard test suite.

For actual Google Drive API integration testing, see test_google_drive.py
For authentication-specific testing, see test_google_drive_authenticate.py
"""

import sys
from pathlib import Path
from unittest import mock
import pytest

from tests.stubs.google_api import install_google_stubs, FakeDriveService

install_google_stubs()

# Add module path for imports
MODULE_DIR = Path(__file__).resolve().parents[2] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))
from GoogleDriveHandler import GoogleDriveHandler




@mock.patch.object(GoogleDriveHandler, '_authenticate', lambda self: setattr(self, 'service', FakeDriveService()))
def test_upload_download_delete(tmp_path):
    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {'root_folder': 'root'}})
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


@mock.patch.object(GoogleDriveHandler, '_authenticate', lambda self: setattr(self, 'service', FakeDriveService()))
def test_list_files(tmp_path):
    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {'root_folder': 'root'}})
    handler.connect()

    (tmp_path / 'src1.txt').write_text('a')
    (tmp_path / 'src2.txt').write_text('b')

    handler.upload_file(str(tmp_path / 'src1.txt'), 'folder/file1.txt')
    handler.upload_file(str(tmp_path / 'src2.txt'), 'folder/file2.txt')

    files = handler.list_files('folder')
    assert sorted(files) == ['file1.txt', 'file2.txt']

