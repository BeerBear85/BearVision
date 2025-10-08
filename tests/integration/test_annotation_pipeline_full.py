import sys
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
import shutil

from tests.stubs import ultralytics  # noqa: F401

MODULE_PATH = Path(__file__).resolve().parents[2] / 'pretraining' / 'annotation'
sys.path.append(str(MODULE_PATH))
import annotation_pipeline as ap


def test_pipeline_full(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    dataset_dir = tmp_path / 'dataset'
    cfg = {
        'videos': ['tests/data/TestMovie1.mp4'],
        'sampling': {'fps': 3.0},
        'quality': {'blur': 0.0, 'luma_min': 0, 'luma_max': 255},
        'yolo': {'weights': 'yolov8s.onnx', 'conf_thr': 0.1},
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

    with mock.patch.object(ap, 'load_config', side_effect=load_cfg):
        ap.main(['run', str(cfg_path)])

    imgs = list((dataset_dir / 'images').glob('*.jpg'))
    lbls = list((dataset_dir / 'labels').glob('*.txt'))
    assert imgs, 'no images generated'
    assert lbls, 'no labels generated'

    debug_path = dataset_dir / 'debug.jsonl'
    content = debug_path.read_text().splitlines()
    assert any('wakeboarder' in line for line in content)

    shutil.rmtree(dataset_dir)
