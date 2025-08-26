#!/usr/bin/env python3
"""
Tests to verify real-time trajectory generation behavior.
These tests ensure trajectories are generated immediately when gaps are detected,
rather than only at the end of video processing.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np

# Add the annotation module path
MODULE_PATH = Path(__file__).resolve().parents[1] / 'pretraining' / 'annotation'
sys.path.append(str(MODULE_PATH))

import annotation_pipeline as ap


def test_realtime_trajectory_generation_with_gaps():
    """Test that trajectories are generated in real-time when gaps are detected."""
    
    # Create temporary directory for test output
    with tempfile.TemporaryDirectory() as tmpdir:
        
        # Track trajectory generation calls with timing information
        trajectory_calls = []
        original_generate = ap.generate_trajectory_during_processing
        
        def mock_generate_trajectory(segment_items, det_points, cfg, track_id, sample_rate):
            """Track when trajectory generation is called during processing."""
            call_info = {
                'track_id': track_id,
                'det_points_count': len(det_points),
                'segment_items_count': len(segment_items),
                'first_frame': segment_items[0]['frame_idx'] if segment_items else None,
                'last_frame': segment_items[-1]['frame_idx'] if segment_items else None,
                'detection_frames': [p[0] for p in det_points]
            }
            trajectory_calls.append(call_info)
            # Return tuple as expected by the function signature
            trajectory_points = [(100 + i, 100 + i) for i in range(len(det_points))]  # Mock trajectory
            final_item = segment_items[-1] if segment_items else None
            return f"{tmpdir}/trajectory_{track_id}.jpg", trajectory_points, final_item
        
        # Mock the trajectory generation function
        ap.generate_trajectory_during_processing = mock_generate_trajectory
        
        try:
            # Create detection pattern: detections in frames 0-30, gap 31-60, detections 61-90
            detection_frames = list(range(0, 31)) + list(range(61, 91))
            
            class MockYOLO:
                def __init__(self):
                    self.frame_count = 0
                    
                def detect(self, frame):
                    result = []
                    if self.frame_count in detection_frames:
                        result = [{
                            'bbox': [100, 100, 150, 150],
                            'cls': 0,
                            'label': 'person',
                            'conf': 0.9
                        }]
                    self.frame_count += 1
                    return result
            
            class MockIngest:
                def __init__(self):
                    self.frame_idx = 0
                    
                def __iter__(self):
                    # Generate 100 frames
                    while self.frame_idx < 100:
                        frame = np.zeros((64, 64, 3), dtype=np.uint8)
                        frame[:] = 128  # Gray frame for quality checks
                        
                        yield {
                            'video': '/fake/video.mp4',
                            'frame_idx': self.frame_idx,
                            'frame': frame
                        }
                        self.frame_idx += 1
            
            # Create test configuration with 1-second gap timeout (30 frames at 30fps)
            cfg = ap.PipelineConfig(
                videos=['/fake/video.mp4'],
                sampling=ap.SamplingConfig(step=1),
                quality=ap.QualityConfig(blur=0, luma_min=0, luma_max=255),
                yolo=ap.YoloConfig(weights='dummy.pt', conf_thr=0.5),
                export=ap.ExportConfig(output_dir=tmpdir, format='dataset'),
                trajectory=ap.TrajectoryConfig(cutoff_hz=0.0),
                detection_gap_timeout_s=1.0  # 1 second = 30 frames gap threshold
            )
            
            # Mock the required objects
            ap.VidIngest = lambda videos, config: MockIngest()
            ap.PreLabelYOLO = lambda config: MockYOLO()
            
            # Run the pipeline
            ap.run(cfg, gui_mode=False)
            
            # Verify real-time trajectory generation behavior
            assert len(trajectory_calls) >= 2, f"Expected at least 2 trajectories, got {len(trajectory_calls)}"
            
            # First trajectory should be generated when gap is detected (around frame 60)
            first_traj = trajectory_calls[0]
            assert first_traj['track_id'] == 1
            assert 0 in first_traj['detection_frames']  # Should include first segment
            assert 30 in first_traj['detection_frames']  # Should include last detection before gap
            assert 61 not in first_traj['detection_frames']  # Should NOT include post-gap detections
            
            # Second trajectory should be generated at end-of-video or when next gap detected
            second_traj = trajectory_calls[1]
            assert second_traj['track_id'] == 2
            assert 61 in second_traj['detection_frames']  # Should include post-gap detections
            assert 90 in second_traj['detection_frames']  # Should include final detections
            assert 30 not in second_traj['detection_frames']  # Should NOT include pre-gap detections
            
        finally:
            # Restore original function
            ap.generate_trajectory_during_processing = original_generate


def test_multiple_gaps_generate_multiple_trajectories():
    """Test that multiple gaps result in multiple trajectory generations."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_calls = []
        original_generate = ap.generate_trajectory_during_processing
        
        def mock_generate_trajectory(segment_items, det_points, cfg, track_id, sample_rate):
            trajectory_calls.append({
                'track_id': track_id,
                'detection_count': len(det_points),
                'detection_frames': [p[0] for p in det_points]
            })
            # Return tuple as expected by the function signature
            trajectory_points = [(100 + i, 100 + i) for i in range(len(det_points))]  # Mock trajectory
            final_item = segment_items[-1] if segment_items else None
            return f"{tmpdir}/trajectory_{track_id}.jpg", trajectory_points, final_item
        
        ap.generate_trajectory_during_processing = mock_generate_trajectory
        
        try:
            # Create pattern: detections 0-20, gap 21-50, detections 51-70, gap 71-100, detections 101-120
            detection_frames = list(range(0, 21)) + list(range(51, 71)) + list(range(101, 121))
            
            class MockYOLO:
                def __init__(self):
                    self.frame_count = 0
                    
                def detect(self, frame):
                    result = []
                    if self.frame_count in detection_frames:
                        result = [{
                            'bbox': [100, 100, 150, 150],
                            'cls': 0,
                            'label': 'person',
                            'conf': 0.9
                        }]
                    self.frame_count += 1
                    return result
            
            class MockIngest:
                def __init__(self):
                    self.frame_idx = 0
                    
                def __iter__(self):
                    while self.frame_idx < 130:
                        frame = np.zeros((64, 64, 3), dtype=np.uint8)
                        frame[:] = 128
                        
                        yield {
                            'video': '/fake/video.mp4',
                            'frame_idx': self.frame_idx,
                            'frame': frame
                        }
                        self.frame_idx += 1
            
            cfg = ap.PipelineConfig(
                videos=['/fake/video.mp4'],
                sampling=ap.SamplingConfig(step=1),
                quality=ap.QualityConfig(blur=0, luma_min=0, luma_max=255),
                yolo=ap.YoloConfig(weights='dummy.pt', conf_thr=0.5),
                export=ap.ExportConfig(output_dir=tmpdir, format='dataset'),
                trajectory=ap.TrajectoryConfig(cutoff_hz=0.0),
                detection_gap_timeout_s=1.0  # 30 frames gap threshold
            )
            
            ap.VidIngest = lambda videos, config: MockIngest()
            ap.PreLabelYOLO = lambda config: MockYOLO()
            
            ap.run(cfg, gui_mode=False)
            
            # Should generate 3 trajectories for 3 segments
            assert len(trajectory_calls) == 3, f"Expected 3 trajectories, got {len(trajectory_calls)}"
            
            # Verify trajectory IDs are sequential
            track_ids = [call['track_id'] for call in trajectory_calls]
            assert track_ids == [1, 2, 3]
            
            # Verify detection frame ranges don't overlap
            traj1_frames = set(trajectory_calls[0]['detection_frames'])
            traj2_frames = set(trajectory_calls[1]['detection_frames'])
            traj3_frames = set(trajectory_calls[2]['detection_frames'])
            
            assert traj1_frames.isdisjoint(traj2_frames), "Trajectory 1 and 2 should not share frames"
            assert traj2_frames.isdisjoint(traj3_frames), "Trajectory 2 and 3 should not share frames"
            assert traj1_frames.isdisjoint(traj3_frames), "Trajectory 1 and 3 should not share frames"
            
        finally:
            ap.generate_trajectory_during_processing = original_generate


def test_no_trajectory_generation_without_gaps():
    """Test that only one trajectory is generated when there are no gaps."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_calls = []
        original_generate = ap.generate_trajectory_during_processing
        
        def mock_generate_trajectory(segment_items, det_points, cfg, track_id, sample_rate):
            trajectory_calls.append({
                'track_id': track_id,
                'detection_count': len(det_points)
            })
            # Return tuple as expected by the function signature
            trajectory_points = [(100 + i, 100 + i) for i in range(len(det_points))]  # Mock trajectory
            final_item = segment_items[-1] if segment_items else None
            return f"{tmpdir}/trajectory_{track_id}.jpg", trajectory_points, final_item
        
        ap.generate_trajectory_during_processing = mock_generate_trajectory
        
        try:
            # Continuous detections with no gaps
            detection_frames = list(range(0, 100))
            
            class MockYOLO:
                def __init__(self):
                    self.frame_count = 0
                    
                def detect(self, frame):
                    result = []
                    if self.frame_count in detection_frames:
                        result = [{
                            'bbox': [100, 100, 150, 150],
                            'cls': 0,
                            'label': 'person',
                            'conf': 0.9
                        }]
                    self.frame_count += 1
                    return result
            
            class MockIngest:
                def __init__(self):
                    self.frame_idx = 0
                    
                def __iter__(self):
                    while self.frame_idx < 100:
                        frame = np.zeros((64, 64, 3), dtype=np.uint8)
                        frame[:] = 128
                        
                        yield {
                            'video': '/fake/video.mp4',
                            'frame_idx': self.frame_idx,
                            'frame': frame
                        }
                        self.frame_idx += 1
            
            cfg = ap.PipelineConfig(
                videos=['/fake/video.mp4'],
                sampling=ap.SamplingConfig(step=1),
                quality=ap.QualityConfig(blur=0, luma_min=0, luma_max=255),
                yolo=ap.YoloConfig(weights='dummy.pt', conf_thr=0.5),
                export=ap.ExportConfig(output_dir=tmpdir, format='dataset'),
                trajectory=ap.TrajectoryConfig(cutoff_hz=0.0),
                detection_gap_timeout_s=1.0
            )
            
            ap.VidIngest = lambda videos, config: MockIngest()
            ap.PreLabelYOLO = lambda config: MockYOLO()
            
            ap.run(cfg, gui_mode=False)
            
            # Should generate only 1 trajectory at end-of-video
            assert len(trajectory_calls) == 1, f"Expected 1 trajectory, got {len(trajectory_calls)}"
            assert trajectory_calls[0]['track_id'] == 1
            assert trajectory_calls[0]['detection_count'] == 100
            
        finally:
            ap.generate_trajectory_during_processing = original_generate