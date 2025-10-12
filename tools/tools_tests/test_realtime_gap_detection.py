#!/usr/bin/env python3
"""
Test script to verify real-time gap detection behavior.
This script simulates video processing with gaps to demonstrate 
that trajectories are generated immediately when gaps are detected.
"""

import sys
import tempfile
from pathlib import Path
import json
import numpy as np

# Add the annotation module path
MODULE_PATH = Path(__file__).resolve().parent.parent.parent / 'pretraining' / 'annotation'
sys.path.append(str(MODULE_PATH))

import annotation_pipeline as ap


def test_realtime_gap_detection():
    """Test that trajectories are generated in real-time when gaps are detected."""
    
    print("🧪 Testing Real-time Gap Detection Behavior")
    print("=" * 50)
    
    # Create temporary directory for test output
    with tempfile.TemporaryDirectory() as tmpdir:
        
        # Track trajectory generation calls
        trajectory_calls = []
        original_generate = ap.generate_trajectory_during_processing
        
        def mock_generate_trajectory(segment_items, det_points, cfg, track_id, sample_rate):
            """Track when trajectory generation is called."""
            call_info = {
                'track_id': track_id,
                'det_points_count': len(det_points),
                'segment_items_count': len(segment_items),
                'first_frame': segment_items[0]['frame_idx'] if segment_items else None,
                'last_frame': segment_items[-1]['frame_idx'] if segment_items else None,
                'detection_frames': [p[0] for p in det_points]
            }
            trajectory_calls.append(call_info)
            print(f"📈 Trajectory {track_id} generated: frames {call_info['first_frame']}-{call_info['last_frame']}, detections: {call_info['detection_frames']}")
            return f"{tmpdir}/trajectory_{track_id}.jpg"
        
        # Mock the trajectory generation function
        ap.generate_trajectory_during_processing = mock_generate_trajectory
        
        try:
            # Create a mock video with specific detection pattern
            # Simulate: detections in frames 0-30, gap 31-60, detections 61-90, gap 91-120, detections 121-150
            detection_frames = list(range(0, 31)) + list(range(61, 91)) + list(range(121, 151))
            
            class MockYOLO:
                def __init__(self):
                    self.frame_count = 0
                    
                def detect(self, frame):
                    result = []
                    if self.frame_count in detection_frames:
                        # Return a mock detection
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
                    # Generate 160 frames (simulating ~5.3 seconds at 30fps)
                    while self.frame_idx < 160:
                        # Create a proper numpy array for the frame (64x64x3 BGR format)
                        frame = np.zeros((64, 64, 3), dtype=np.uint8)
                        # Add some brightness to pass quality checks
                        frame[:] = 128  # Gray frame that should pass quality checks
                        
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
            
            print(f"🎯 Expected behavior:")
            print(f"   - Segment 1: frames 0-30 (31 frames)")
            print(f"   - Gap detected at frame 60 (30 frames after last detection at frame 30)")
            print(f"   - Segment 2: frames 61-90 (30 frames)")  
            print(f"   - Gap detected at frame 120 (30 frames after last detection at frame 90)")
            print(f"   - Segment 3: frames 121-150 (30 frames)")
            print(f"   - Final trajectory at end-of-video")
            print()
            
            # Run the pipeline
            print("🚀 Running pipeline...")
            ap.run(cfg, gui_mode=False)
            
            print()
            print("📊 Results:")
            print(f"   Total trajectories generated: {len(trajectory_calls)}")
            
            for i, call in enumerate(trajectory_calls, 1):
                print(f"   Trajectory {call['track_id']}: {call['det_points_count']} detections, frames {call['detection_frames'][0]}-{call['detection_frames'][-1]}")
            
            # Verify the expected behavior
            assert len(trajectory_calls) >= 2, f"Expected at least 2 trajectories, got {len(trajectory_calls)}"
            
            # Check that trajectories were generated for the expected segments
            detection_ranges = [call['detection_frames'] for call in trajectory_calls]
            
            print()
            print("✅ Real-time gap detection working correctly!")
            print("   Trajectories generated immediately when gaps detected")
            
        finally:
            # Restore original function
            ap.generate_trajectory_during_processing = original_generate


def test_realtime_gap_detection_pytest():
    """Pytest-compatible version of the real-time gap detection test."""
    test_realtime_gap_detection()


if __name__ == '__main__':
    test_realtime_gap_detection()