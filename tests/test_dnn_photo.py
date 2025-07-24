import sys
from pathlib import Path
from unittest import mock

import cv2

MODULE_PATH = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_PATH))
from DnnHandler import DnnHandler


def test_dnn_photo_person_detection():
    image_dir = Path('test/images')
    image_paths = [image_dir / f'test_image_{i}.jpg' for i in range(1, 6)]

    for p in image_paths:
        assert p.is_file(), f'Missing {p}'

    handler = DnnHandler('yolov8s')
    with mock.patch.object(handler, 'init', return_value=None) as init_mock, \
         mock.patch.object(handler, 'find_person', return_value=([[1, 2, 3, 4]], [0.9])) as find_mock:
        handler.init()
        for p in image_paths:
            frame = cv2.imread(str(p))
            handler.find_person(frame)

    init_mock.assert_called_once()
    assert find_mock.call_count == len(image_paths)
