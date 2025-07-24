import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
import shutil

# Ensure the annotation module can be imported without ultralytics installed
sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=lambda *a, **k: None)

MODULE_PATH = Path(__file__).resolve().parents[1] / 'pretraining' / 'annotation'
sys.path.append(str(MODULE_PATH))
import annotation_pipeline as ap


def test_pipeline_full(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    dataset_dir = tmp_path / 'dataset'
    cfg = {
        'videos': ['test/input_video/TestMovie1.mp4'],
        'sampling': {'fps': 1.0},
        'quality': {'blur': 0.0, 'luma_min': 0, 'luma_max': 255},
        'yolo': {'weights': 'yolov8n.pt', 'conf_thr': 0.1},
        'export': {'output_dir': str(dataset_dir)},
    }
    cfg_path.write_text(yaml.dump(cfg))

    def load_cfg(_):
        data = yaml.safe_load(cfg_path.read_text())
        return {
            'videos': data.get('videos', []),
            'sampling': ap.SamplingConfig(**data.get('sampling', {})),
            'quality': ap.QualityConfig(**data.get('quality', {})),
            'yolo': ap.YoloConfig(**data.get('yolo', {})),
            'export': ap.ExportConfig(**data.get('export', {})),
        }

    class DummyBox:
        def __init__(self):
            self.xyxy = [np.array([0, 0, 10, 10], dtype=float)]
            self.conf = [np.array([0.9], dtype=float)]
            self.cls = [np.array([0], dtype=float)]

    class DummyResults:
        def __init__(self):
            self.boxes = [DummyBox()]

    class DummyModel:
        def __init__(self, weights):
            self.names = {0: 'person'}
        def __call__(self, frame):
            return [DummyResults()]

    with mock.patch.object(ap, 'YOLO', DummyModel), \
         mock.patch.object(ap, 'load_config', side_effect=load_cfg):
        ap.main(['run', str(cfg_path)])

    imgs = list((dataset_dir / 'images').glob('*.jpg'))
    lbls = list((dataset_dir / 'labels').glob('*.txt'))
    assert imgs, 'no images generated'
    assert lbls, 'no labels generated'

    shutil.rmtree(dataset_dir)
