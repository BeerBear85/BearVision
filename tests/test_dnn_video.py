import sys
from pathlib import Path
from unittest import mock

import cv2

MODULE_PATH = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_PATH))
from DnnHandler import DnnHandler


def test_dnn_video_person_detection():
    video_path = Path('tests/data/TestMovie1.mp4')
    assert video_path.is_file(), f'Missing {video_path}'

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    assert ret, 'Failed to read first frame'

    handler = DnnHandler('yolov8s')
    with mock.patch.object(handler, 'init', return_value=None) as init_mock, \
         mock.patch.object(handler, 'find_person', return_value=([[1, 2, 3, 4]], [0.9])) as find_mock:
        handler.init()
        boxes, confidences = handler.find_person(frame)

    init_mock.assert_called_once()
    find_mock.assert_called_once_with(frame)
    assert boxes == [[1, 2, 3, 4]]
    assert confidences == [0.9]
