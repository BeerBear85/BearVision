from pathlib import Path

import pytest


cv2 = pytest.importorskip("cv2")
opencv_dnn = pytest.importorskip("bearvision.integrations.opencv_dnn")
DnnHandler = opencv_dnn.DnnHandler


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "code/dnn_models/yolov8n.onnx"


@pytest.fixture(scope="module")
def detector():
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    handler = DnnHandler(str(MODEL))
    handler.confidence_threshold = 0.35
    handler.init()
    return handler


@pytest.mark.parametrize(
    ("relative_path", "minimum_people"),
    [
        ("tests/end2end/images/test_image_1.jpg", 1),
        ("tests/end2end/images/test_image_2.jpg", 1),
        ("tests/end2end/images/test_image_3.jpg", 1),
        ("tests/end2end/images/test_image_4.jpg", 2),
        ("tests/end2end/images/test_image_5.jpg", 1),
        ("tests/end2end/images/test_image_easy.jpg", 1),
    ],
)
def test_checked_in_images_still_detect_people(
    detector,
    relative_path: str,
    minimum_people: int,
) -> None:
    image = cv2.imread(str(ROOT / relative_path))
    assert image is not None, f"could not read {relative_path}"

    boxes, confidences = detector.find_person(image)

    assert len(boxes) >= minimum_people
    assert len(confidences) == len(boxes)


def test_easy_reference_image_remains_high_confidence(detector) -> None:
    image = cv2.imread(str(ROOT / "tests/end2end/images/test_image_easy.jpg"))

    _, confidences = detector.find_person(image)

    assert max(confidences) >= 0.85
