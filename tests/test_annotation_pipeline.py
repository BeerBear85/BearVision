import json
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tests.stubs import ultralytics  # noqa: F401

MODULE_PATH = Path(__file__).resolve().parents[1] / 'pretraining' / 'annotation'
sys.path.append(str(MODULE_PATH))
import annotation_pipeline as ap


def create_dummy_video(path, num_frames=5, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(num_frames):
        frame = np.full((64, 64, 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_load_config(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('videos:\n  - a.mp4\nsampling:\n  fps: 2.0\n')
    loaded = ap.load_config(str(cfg_path))
    assert loaded == {'videos': ['a.mp4'], 'sampling': {'fps': 2.0}}


def test_vid_ingest_fps(tmp_path):
    video = create_dummy_video(tmp_path / 'v.mp4')
    ingest = ap.VidIngest([str(video)], ap.SamplingConfig(fps=2))
    frames = list(ingest)
    assert [f['frame_idx'] for f in frames] == [0, 2, 4]


def test_quality_filter():
    cfg = ap.QualityConfig(blur=10, luma_min=50, luma_max=200)
    qf = ap.QualityFilter(cfg)
    good = np.zeros((10, 10, 3), dtype=np.uint8)
    good[5:, 5:] = 255
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    assert qf.check(good)
    assert not qf.check(bad)


def test_prelabel_yolo_detection():
    class DummyBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = [np.array(xyxy, dtype=float)]
            self.conf = [np.array([conf], dtype=float)]
            self.cls = [np.array([cls], dtype=float)]

    class DummyResults:
        def __init__(self, boxes):
            self.boxes = boxes

    class DummyModel:
        def __init__(self, boxes):
            self.boxes = boxes
            self.names = {0: 'person', 1: 'car'}

        def __call__(self, frame):
            return [DummyResults(self.boxes)]

    d_boxes = [DummyBox([0, 0, 10, 10], 0.8, 0), DummyBox([0, 0, 5, 5], 0.9, 1)]
    with mock.patch.object(ap, 'YOLO', return_value=DummyModel(d_boxes)):
        yolo = ap.PreLabelYOLO(ap.YoloConfig(weights='x.pt', conf_thr=0.5))
        out = yolo.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out == [
        {
            'bbox': [0, 0, 10, 10],
            'cls': 0,
            'label': 'person',
            'conf': 0.8,
        }
    ]


def test_prelabel_with_dnn_handler():
    with mock.patch.object(ap, 'DnnHandler') as dh_mock:
        instance = dh_mock.return_value
        instance.init.return_value = None
        instance.find_person.return_value = ([[1, 2, 3, 4]], [0.9])
        yolo = ap.PreLabelYOLO(ap.YoloConfig(weights='yolov8s.onnx', conf_thr=0.5))
        out = yolo.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out == [
        {
            'bbox': [1, 2, 4, 6],
            'cls': 0,
            'label': 'wakeboarder',
            'conf': 0.9,
        }
    ]


def test_dataset_exporter(tmp_path):
    exporter = ap.DatasetExporter(ap.ExportConfig(output_dir=str(tmp_path)))
    item = {'frame': np.zeros((4, 4, 3), dtype=np.uint8), 'frame_idx': 1, 'video': str(tmp_path / 'vid.mp4')}
    boxes = [{'bbox': [0, 0, 2, 2], 'cls': 0, 'conf': 1.0}]
    exporter.save(item, boxes)
    exporter.close()

    img_files = list((tmp_path / 'images').iterdir())
    lbl_files = list((tmp_path / 'labels').iterdir())
    assert len(img_files) == 1
    assert len(lbl_files) == 1

    lbl_content = lbl_files[0].read_text().strip()
    assert lbl_content.startswith('0 ')

    debug_path = tmp_path / 'debug.jsonl'
    lines = debug_path.read_text().strip().split('\n')
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec['frame_idx'] == 1


def test_cvat_exporter(tmp_path):
    exporter = ap.CvatExporter(ap.ExportConfig(output_dir=str(tmp_path), format="cvat"))
    item = {
        'frame': np.zeros((4, 4, 3), dtype=np.uint8),
        'frame_idx': 1,
        'video': str(tmp_path / 'vid.mp4'),
    }
    boxes = [{'bbox': [0, 0, 2, 2], 'cls': 0, 'label': 'cls0', 'conf': 1.0}]
    exporter.save(item, boxes)
    exporter.close()

    img_files = list((tmp_path / 'images').iterdir())
    assert len(img_files) == 1
    xml_path = tmp_path / 'annotations.xml'
    assert xml_path.exists()
    content = xml_path.read_text()
    assert '<image' in content and '<box' in content


def test_run_skips_frames_without_detections(tmp_path):
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=2)
    cfg = {
        'videos': [str(video)],
        'sampling': ap.SamplingConfig(),
        'quality': ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        'yolo': ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        'export': ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    }
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.touch()
    with mock.patch.object(ap, 'load_config', return_value=cfg):
        with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
            MockYolo.return_value.detect.return_value = []
            ap.run(str(cfg_path))
    dataset_dir = tmp_path / 'dataset'
    imgs = list((dataset_dir / 'images').glob('*'))
    lbls = list((dataset_dir / 'labels').glob('*'))
    assert not imgs
    assert not lbls


def test_run_with_preview(tmp_path):
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=1)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    )
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo, \
         mock.patch('cv2.imshow') as imshow_mock, \
         mock.patch('cv2.waitKey', return_value=-1), \
         mock.patch('cv2.destroyAllWindows'):
        MockYolo.return_value.detect.return_value = [
            {'bbox': [0, 0, 2, 2], 'cls': 0, 'conf': 1.0}
        ]
        ap.run(cfg, show_preview=True)
    dataset_dir = tmp_path / 'dataset'
    imgs = list((dataset_dir / 'images').glob('*.jpg'))
    lbls = list((dataset_dir / 'labels').glob('*.txt'))
    assert imgs and lbls
    assert imshow_mock.called
