import json
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

# Ensure the annotation module can be imported without ultralytics installed
import types
sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=lambda *a, **k: None)

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
        def __call__(self, frame):
            return [DummyResults(self.boxes)]

    d_boxes = [DummyBox([0, 0, 10, 10], 0.8, 1), DummyBox([0, 0, 5, 5], 0.2, 2)]
    with mock.patch.object(ap, 'YOLO', return_value=DummyModel(d_boxes)):
        yolo = ap.PreLabelYOLO(ap.YoloConfig(weights='x.pt', conf_thr=0.5))
        out = yolo.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out == [
        {
            'bbox': [0, 0, 10, 10],
            'cls': 1,
            'label': '1',
            'conf': 0.8,
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
