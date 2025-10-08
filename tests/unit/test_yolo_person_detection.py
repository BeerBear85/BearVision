import sys
from pathlib import Path
from unittest import mock

import cv2

MODULE_PATH = Path(__file__).resolve().parents[2] / 'code' / 'modules'
sys.path.append(str(MODULE_PATH))
from DnnHandler import DnnHandler


def test_highgrade_yolo_person_detection():
    img_path = Path('test/images/test_image_1.jpg')
    frame = cv2.imread(str(img_path))
    handler = DnnHandler('yolov8x')
    with mock.patch.object(handler, 'init', return_value=None), \
         mock.patch.object(handler, 'find_person', return_value=([[10, 20, 30, 40]], [0.9])):
        handler.init()
        boxes, confidences = handler.find_person(frame)
    assert boxes == [[10, 20, 30, 40]]
    assert confidences == [0.9]
