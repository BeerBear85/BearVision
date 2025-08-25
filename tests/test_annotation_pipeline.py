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


def create_dummy_video(path, num_frames=5, fps=5, size=(64, 64)):
    """Create a synthetic video for testing.

    Purpose
    -------
    Produce deterministic pixel data so tests can reason about exact
    coordinates without relying on external assets.

    Inputs
    ------
    path: PathLike
        Destination of the video file.
    num_frames: int, default ``5``
        Number of frames to encode.
    fps: int, default ``5``
        Frame rate of the generated video.
    size: tuple[int, int], default ``(64, 64)``
        Width and height of each frame.

    Outputs
    -------
    Path
        Location of the written video.
    """

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    w, h = size
    for i in range(num_frames):
        # Using ``np.full`` ensures predictable content across all frames.
        frame = np.full((h, w, 3), i, dtype=np.uint8)
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


def test_filter_small_person_boxes_rejects():
    """Tiny person detection should be discarded."""
    boxes = [{'bbox': [0, 0, 1, 1], 'cls': 0, 'label': 'person'}]
    kept, discarded = ap.filter_small_person_boxes(boxes, (2000, 2000), 0.001)
    assert not kept
    assert discarded and discarded[0]['bbox'] == [0, 0, 1, 1]


def test_filter_small_person_boxes_accepts():
    """Normal-sized detections remain untouched."""
    boxes = [{'bbox': [0, 0, 100, 100], 'cls': 0, 'label': 'person'}]
    kept, discarded = ap.filter_small_person_boxes(boxes, (200, 200), 0.001)
    assert kept == boxes
    assert not discarded


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


def test_run_filters_tiny_person_detections(tmp_path):
    """Detections smaller than the diagonal ratio are removed."""
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=1, size=(2000, 2000))
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    )
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.return_value = [
            {'bbox': [0, 0, 1, 1], 'cls': 0, 'label': 'person', 'conf': 1.0}
        ]
        ap.run(cfg)
    dataset_dir = tmp_path / 'dataset'
    imgs = list((dataset_dir / 'images').glob('*'))
    lbls = list((dataset_dir / 'labels').glob('*'))
    assert not imgs and not lbls
    recs = [json.loads(l) for l in (dataset_dir / 'debug.jsonl').read_text().splitlines()]
    assert recs and recs[0]['discarded_boxes']


def test_run_respects_custom_min_diagonal_ratio(tmp_path):
    """Changing the ratio allows keeping otherwise tiny detections."""
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=1, size=(2000, 2000))
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5, min_person_bbox_diagonal_ratio=0.0),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    )
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.return_value = [
            {'bbox': [0, 0, 1, 1], 'cls': 0, 'label': 'person', 'conf': 1.0}
        ]
        ap.run(cfg)
    dataset_dir = tmp_path / 'dataset'
    imgs = list((dataset_dir / 'images').glob('*'))
    lbls = list((dataset_dir / 'labels').glob('*'))
    assert imgs and lbls
    recs = [json.loads(l) for l in (dataset_dir / 'debug.jsonl').read_text().splitlines()]
    assert not recs[0].get('discarded_boxes')


def test_status_reports_total_and_processed_frames(tmp_path):
    """Status should expose total frames and live processed count."""
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=4)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(step=2),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    )
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.return_value = []
        frames = []
        ap.run(cfg, frame_callback=lambda f: frames.append(f))

    # Total frames reflect video metadata; processed_frame_count counts sampled frames.
    # current_frame reflects the last video frame index processed.
    assert ap.status.total_frames == 4
    assert ap.status.processed_frame_count == 2
    assert ap.status.current_frame == 2  # Last processed frame index (0-based)
    assert len(frames) == 2


def test_run_with_preview(tmp_path):
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=1)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
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


def test_interpolation_generates_missing_frames(tmp_path):
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=3)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    )

    call_idx = {'i': 0}

    def side_effect(frame):
        idx = call_idx['i']
        call_idx['i'] += 1
        if idx == 0:
            return [{'bbox': [0, 0, 10, 10], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        if idx == 2:
            return [{'bbox': [20, 20, 30, 30], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        return []

    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.side_effect = side_effect
        ap.run(cfg)

    dataset_dir = tmp_path / 'dataset'
    imgs = sorted((dataset_dir / 'images').glob('*.jpg'))
    lbls = sorted((dataset_dir / 'labels').glob('*.txt'))
    assert len(imgs) == 3
    assert len(lbls) == 3
    mid_vals = lbls[1].read_text().strip().split()
    # Center should lie midway between the two detections
    assert abs(float(mid_vals[1]) - 15) < 1e-3
    assert abs(float(mid_vals[2]) - 15) < 1e-3


def test_interpolation_waits_for_keypress(tmp_path):
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=3)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    )

    call_idx = {'i': 0}

    def side_effect(frame):
        idx = call_idx['i']
        call_idx['i'] += 1
        if idx == 0:
            return [{'bbox': [0, 0, 10, 10], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        if idx == 2:
            return [{'bbox': [20, 20, 30, 30], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        return []

    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo, \
         mock.patch('cv2.imshow'), \
         mock.patch('cv2.waitKey', return_value=-1) as wait_mock, \
         mock.patch('cv2.destroyAllWindows'):
        MockYolo.return_value.detect.side_effect = side_effect
        ap.run(cfg, show_preview=True)

    # The last waitKey call should be with 0, indicating the pause on the final
    # frame awaiting user confirmation.
    assert wait_mock.call_args_list[-1][0][0] == 0


def test_split_trajectories_on_detection_gap(tmp_path):
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=40, fps=5)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
    )

    def side_effect(frame):
        idx = side_effect.call_idx
        side_effect.call_idx += 1
        if idx in (0, 1):
            return [{'bbox': [0, 0, 10, 10], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        if idx in (25, 26):
            return [{'bbox': [20, 20, 30, 30], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        return []

    side_effect.call_idx = 0

    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.side_effect = side_effect
        ap.run(cfg)

    debug_path = tmp_path / 'dataset' / 'debug.jsonl'
    records = [json.loads(l) for l in debug_path.read_text().splitlines()]
    track_ids = {b['track_id'] for r in records for b in r['labels']}
    assert track_ids == {1, 2}
    track1_frames = [r['frame_idx'] for r in records if r['labels'][0]['track_id'] == 1]
    assert max(track1_frames) == 16
    last_track1 = max(
        (r for r in records if r['labels'][0]['track_id'] == 1),
        key=lambda r: r['frame_idx'],
    )
    assert last_track1['labels'][0]['conf'] == 0.0


def test_first_detection_starts_trajectory_segment(tmp_path):
    """First detection in video should always start a new trajectory segment."""
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=10, fps=5)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
        detection_gap_timeout_s=3.0,
    )

    def side_effect(frame):
        idx = side_effect.call_idx
        side_effect.call_idx += 1
        # First detection at frame 5 (after some frames without detections)
        if idx == 5:
            return [{'bbox': [10, 10, 20, 20], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        return []

    side_effect.call_idx = 0

    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.side_effect = side_effect
        ap.run(cfg)

    debug_path = tmp_path / 'dataset' / 'debug.jsonl'
    records = [json.loads(l) for l in debug_path.read_text().splitlines()]
    
    # Should have at least one record with a track_id (trajectory started)
    assert any('labels' in r and r['labels'] and 'track_id' in r['labels'][0] for r in records)
    
    # The first detection should have track_id 1 (first segment)
    first_detection_record = next(r for r in records if r['frame_idx'] == 5)
    assert first_detection_record['labels'][0]['track_id'] == 1


def test_riders_detected_counter_starts_at_zero():
    """Test that rider counter initializes to zero in PipelineStatus."""
    status = ap.PipelineStatus()
    assert status.riders_detected == 0


def test_riders_detected_counter_increments_on_segment_detection(tmp_path):
    """Test that the rider counter increments when rider segments are detected."""
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=10, fps=5)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
        detection_gap_timeout_s=3.0,
    )

    def side_effect(frame):
        idx = side_effect.call_idx
        side_effect.call_idx += 1
        # Single detection that will form one segment
        if idx == 1:
            return [{'bbox': [10, 10, 20, 20], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        return []

    side_effect.call_idx = 0

    # Reset status for clean test
    ap.status = ap.PipelineStatus()
    
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.side_effect = side_effect
        ap.run(cfg)

    # Should have detected 1 rider segment
    assert ap.status.riders_detected == 1


def test_riders_detected_counter_increments_multiple_segments(tmp_path):
    """Test that the rider counter increments for multiple separate rider segments."""
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=40, fps=5)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
        detection_gap_timeout_s=3.0,
    )

    def side_effect(frame):
        idx = side_effect.call_idx
        side_effect.call_idx += 1
        # Two separate segments with gap between them
        if idx in (1, 2):  # First segment
            return [{'bbox': [10, 10, 20, 20], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        if idx in (25, 26):  # Second segment after gap
            return [{'bbox': [30, 30, 40, 40], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        return []

    side_effect.call_idx = 0

    # Reset status for clean test
    ap.status = ap.PipelineStatus()
    
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.side_effect = side_effect
        ap.run(cfg)

    # Should have detected 2 rider segments
    assert ap.status.riders_detected == 2


def test_riders_detected_counter_logs_message(tmp_path, caplog):
    """Test that rider detection logs the counter message."""
    import logging
    caplog.set_level(logging.INFO)
    
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=10, fps=5)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
        detection_gap_timeout_s=3.0,
    )

    def side_effect(frame):
        idx = side_effect.call_idx
        side_effect.call_idx += 1
        if idx == 1:
            return [{'bbox': [10, 10, 20, 20], 'cls': 0, 'label': 'person', 'conf': 1.0}]
        return []

    side_effect.call_idx = 0

    # Reset status for clean test
    ap.status = ap.PipelineStatus()
    
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.side_effect = side_effect
        ap.run(cfg)

    # Check that the log message was emitted
    assert "Riders detected: 1" in caplog.text


def test_riders_detected_counter_resets_on_new_session(tmp_path):
    """Test that the rider counter resets when status is reset."""
    # First, set up a status with some riders detected
    ap.status = ap.PipelineStatus()
    ap.status.riders_detected = 5
    
    # Reset status (simulating what happens when GUI starts a new session)
    ap.status = ap.PipelineStatus()
    
    # Should be back to 0
    assert ap.status.riders_detected == 0


def test_riders_detected_counter_no_detections(tmp_path):
    """Test that rider counter remains 0 when no riders are detected."""
    video = create_dummy_video(tmp_path / 'v.mp4', num_frames=10, fps=5)
    cfg = ap.PipelineConfig(
        videos=[str(video)],
        sampling=ap.SamplingConfig(fps=5),
        quality=ap.QualityConfig(blur=0.0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.onnx', conf_thr=0.1),
        export=ap.ExportConfig(output_dir=str(tmp_path / 'dataset')),
        detection_gap_timeout_s=3.0,
    )

    # Reset status for clean test
    ap.status = ap.PipelineStatus()
    
    with mock.patch.object(ap, 'PreLabelYOLO') as MockYolo:
        MockYolo.return_value.detect.return_value = []  # No detections
        ap.run(cfg)

    # Should remain 0 since no riders were detected
    assert ap.status.riders_detected == 0
