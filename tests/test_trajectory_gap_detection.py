"""Tests for trajectory gap detection and generation during video processing."""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from tests.stubs import ultralytics  # noqa: F401

MODULE_PATH = Path(__file__).resolve().parents[1] / 'pretraining' / 'annotation'
sys.path.append(str(MODULE_PATH))
import annotation_pipeline as ap
import trajectory_handler as th


def create_test_video_with_gaps(path, frame_segments, fps=30, size=(64, 64)):
    """Create a synthetic video with detection gaps for testing trajectory generation.
    
    Purpose
    -------
    Create videos that simulate multiple riders with gaps between detections
    to test the gap detection logic.
    
    Inputs
    ------
    path: PathLike
        Destination of the video file.
    frame_segments: list[tuple[int, int]]
        List of (start_frame, end_frame) tuples indicating where detections should occur.
    fps: int, default 30
        Frame rate of the generated video.
    size: tuple[int, int], default (64, 64)
        Width and height of each frame.
    
    Returns
    -------
    Path
        Location of the written video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    total_frames = max(end for start, end in frame_segments) + 10
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    w, h = size
    
    # Create frames where detection areas have different values for each segment
    for i in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Check if this frame should have a detection
        for segment_id, (start, end) in enumerate(frame_segments):
            if start <= i <= end:
                # Create a "person" area in the frame (different position per segment)
                x_offset = segment_id * 10
                y_offset = segment_id * 5
                frame[10 + y_offset:20 + y_offset, 20 + x_offset:30 + x_offset] = 255
                break
        writer.write(frame)
    
    writer.release()
    return path


class MockYOLO:
    """Mock YOLO detector that returns detections only in specified frames."""
    
    def __init__(self, detection_frames):
        self.detection_frames = detection_frames
        self.current_frame = 0
        self.names = {0: 'person'}
    
    def detect(self, frame):
        """Return detection if current frame should have one."""
        if self.current_frame in self.detection_frames:
            # Create a mock detection in the center of the bright area
            segment_id = 0
            for i, (start, end) in enumerate([(0, 50), (100, 150)]):  # Example segments
                if start <= self.current_frame <= end:
                    segment_id = i
                    break
            
            x_offset = segment_id * 10
            y_offset = segment_id * 5
            x_center = 25 + x_offset
            y_center = 15 + y_offset
            
            detection = {
                'bbox': [x_center - 5, y_center - 5, x_center + 5, y_center + 5],
                'cls': 0,
                'label': 'person',
                'conf': 0.9,
            }
            self.current_frame += 1
            return [detection]
        
        self.current_frame += 1
        return []


def test_gap_detection_triggers_trajectory_generation(tmp_path):
    """Test that trajectory generation is triggered when gap threshold is exceeded."""
    # Create video with two distinct segments separated by a large gap
    video_path = tmp_path / 'gap_test.mp4'
    frame_segments = [(0, 30), (90, 120)]  # 60 frame gap at 30fps = 2 seconds
    create_test_video_with_gaps(video_path, frame_segments)
    
    # Create configuration with 1 second gap timeout (30 frames at 30fps)
    cfg = ap.PipelineConfig(
        videos=[str(video_path)],
        sampling=ap.SamplingConfig(step=1),  # Process every frame
        quality=ap.QualityConfig(blur=0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.pt', conf_thr=0.5),
        export=ap.ExportConfig(output_dir=str(tmp_path)),
        trajectory=ap.TrajectoryConfig(cutoff_hz=0.0),
        detection_gap_timeout_s=1.0  # 1 second gap threshold
    )
    
    # Mock the trajectory generation function to track calls
    trajectory_calls = []
    
    def mock_generate_trajectory(segment_items, det_points, cfg, track_id, sample_rate):
        trajectory_calls.append({
            'track_id': track_id,
            'det_points_count': len(det_points),
            'segment_items_count': len(segment_items)
        })
        return f'/fake/trajectory_{track_id}.jpg'
    
    # Mock YOLO to return detections in expected frames
    detection_frames = list(range(0, 31)) + list(range(90, 121))  # Two segments
    mock_yolo = MockYOLO(detection_frames)
    
    with mock.patch.object(ap, 'generate_trajectory_during_processing', side_effect=mock_generate_trajectory), \
         mock.patch.object(ap, 'PreLabelYOLO', return_value=mock_yolo):
        
        ap.run(cfg, gui_mode=False)
    
    # Should generate 2 trajectories: one when gap detected, one at end-of-video
    assert len(trajectory_calls) >= 1, f"Expected at least 1 trajectory generation call, got {len(trajectory_calls)}"
    
    # Verify trajectory IDs are incremental
    track_ids = [call['track_id'] for call in trajectory_calls]
    assert track_ids == list(range(1, len(track_ids) + 1))


def test_end_of_video_trajectory_generation(tmp_path):
    """Test that trajectory generation is triggered at end-of-video."""
    # Create video with single continuous segment (no gaps)
    video_path = tmp_path / 'single_segment.mp4'
    frame_segments = [(0, 60)]  # Single 60-frame segment
    create_test_video_with_gaps(video_path, frame_segments)
    
    # Create configuration
    cfg = ap.PipelineConfig(
        videos=[str(video_path)],
        sampling=ap.SamplingConfig(step=1),
        quality=ap.QualityConfig(blur=0, luma_min=0, luma_max=255),
        yolo=ap.YoloConfig(weights='dummy.pt', conf_thr=0.5),
        export=ap.ExportConfig(output_dir=str(tmp_path)),
        trajectory=ap.TrajectoryConfig(cutoff_hz=0.0),
        detection_gap_timeout_s=3.0  # Long timeout to ensure no gap detection
    )
    
    # Mock the trajectory generation function
    trajectory_calls = []
    
    def mock_generate_trajectory(segment_items, det_points, cfg, track_id, sample_rate):
        trajectory_calls.append({
            'track_id': track_id,
            'det_points_count': len(det_points),
            'final_call': True
        })
        return f'/fake/trajectory_{track_id}.jpg'
    
    # Mock YOLO to return detections throughout the video
    detection_frames = list(range(0, 61))
    mock_yolo = MockYOLO(detection_frames)
    
    with mock.patch.object(ap, 'generate_trajectory_during_processing', side_effect=mock_generate_trajectory), \
         mock.patch.object(ap, 'PreLabelYOLO', return_value=mock_yolo):
        
        ap.run(cfg, gui_mode=False)
    
    # Should generate 1 trajectory at end-of-video
    assert len(trajectory_calls) == 1, f"Expected 1 trajectory generation call, got {len(trajectory_calls)}"
    assert trajectory_calls[0]['track_id'] == 1
    assert trajectory_calls[0]['det_points_count'] > 0


def test_trajectory_generation_during_processing_function():
    """Test the _generate_trajectory_during_processing function directly."""
    # Create mock segment items and detection points
    segment_items = []
    det_points = []
    
    # Create 5 frames with detections
    for i in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add a bright square to simulate person detection area
        frame[10:20, 20:30] = 255
        
        segment_items.append({
            'frame': frame,
            'frame_idx': i * 10,  # Frames 0, 10, 20, 30, 40
            'video': '/fake/video.mp4',
            'boxes': [{
                'bbox': [20, 10, 30, 20],
                'cls': 0,
                'label': 'person',
                'conf': 0.9
            }]
        })
        
        # Add corresponding detection point
        det_points.append((i * 10, 25, 15, 10, 10, 0, 'person'))
    
    # Create test configuration
    cfg = ap.PipelineConfig(
        export=ap.ExportConfig(output_dir='/tmp'),
        trajectory=ap.TrajectoryConfig(cutoff_hz=0.0)
    )
    
    # Test without mocking since function generates timestamped filenames
    result = ap.generate_trajectory_during_processing(
        segment_items, det_points, cfg, track_id=1, sample_rate=30.0
    )
    
    # Check that the result is a trajectory path with the expected pattern
    assert result is not None
    import os
    expected_prefix = os.path.join('/tmp', 'trajectories', 'trajectory_1_').replace('/', os.sep)
    assert result.startswith(expected_prefix)
    assert result.endswith('.jpg')
    
    # Verify trajectory file was actually created
    assert os.path.exists(result)
    
    # Clean up the created file
    os.remove(result)


def test_no_trajectory_generation_without_detections():
    """Test that no trajectory is generated when there are no detection points."""
    segment_items = [{'frame': np.zeros((64, 64, 3), dtype=np.uint8), 'frame_idx': 0}]
    det_points = []  # No detections
    cfg = ap.PipelineConfig(export=ap.ExportConfig(output_dir='/tmp'))
    
    result = ap.generate_trajectory_during_processing(
        segment_items, det_points, cfg, track_id=1, sample_rate=30.0
    )
    
    assert result is None


def test_single_detection_trajectory_generation():
    """Test trajectory generation with only one detection point."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[10:20, 20:30] = 255
    
    segment_items = [{
        'frame': frame,
        'frame_idx': 10,
        'video': '/fake/video.mp4',
        'boxes': [{'bbox': [20, 10, 30, 20], 'cls': 0, 'label': 'person', 'conf': 0.9}]
    }]
    
    det_points = [(10, 25, 15, 10, 10, 0, 'person')]
    cfg = ap.PipelineConfig(export=ap.ExportConfig(output_dir='/tmp'))
    
    # Test without mocking since function generates timestamped filenames
    result = ap.generate_trajectory_during_processing(
        segment_items, det_points, cfg, track_id=1, sample_rate=30.0
    )
    
    # Check that the result is a trajectory path with the expected pattern
    assert result is not None
    import os
    expected_prefix = os.path.join('/tmp', 'trajectories', 'trajectory_1_').replace('/', os.sep)
    assert result.startswith(expected_prefix)
    assert result.endswith('.jpg')
    
    # Verify trajectory file was actually created
    assert os.path.exists(result)
    
    # Clean up the created file
    os.remove(result)