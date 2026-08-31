#!/usr/bin/env python3
"""
Test EdgeApplication detection worker in isolation.
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
APP_DIR = Path(__file__).resolve().parents[2] / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

from edge_application import EdgeApplication, SystemStatus

def test_detection_worker_isolation():
    """Test the detection worker thread in isolation."""
    print("=== Detection Worker Isolation Test ===")

    frames_received = []

    def frame_callback(frame: np.ndarray):
        frames_received.append(frame.shape)
        print(f"FRAME RECEIVED: {frame.shape}")

    def log_callback(level: str, message: str):
        print(f"[{level.upper()}] {message}")

    def status_callback(status: SystemStatus):
        print(f"Status: {status.overall_status.value}")

    try:
        # Create EdgeApplication with callbacks
        edge_app = EdgeApplication(
            status_callback=status_callback,
            log_callback=log_callback,
            frame_callback=frame_callback
        )

        # Initialize basic components (but skip GoPro connection)
        edge_app.initialized = True
        edge_app.running = True
        edge_app.status_manager.update_status(preview_active=True)

        # Set up stream processor if available
        if hasattr(edge_app.system_coordinator, 'stream_processor') and edge_app.system_coordinator.stream_processor:
            edge_app.system_coordinator.stream_processor.set_preview_stream_url("udp://172.24.106.51:8554")

        print("1. Starting detection/stream processing...")

        # Try to start stream processing
        if hasattr(edge_app.system_coordinator, 'stream_processor') and edge_app.system_coordinator.stream_processor:
            success = edge_app.system_coordinator.stream_processor.start_processing()
            print(f"Stream processor started: {success}")
        else:
            print("Stream processor not available - skipping test")
            return False

        print("2. Waiting for frames (15 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 15:
            time.sleep(0.5)
            if len(frames_received) >= 3:
                break

        print("3. Stopping worker...")
        edge_app.running = False
        if hasattr(edge_app.system_coordinator, 'stream_processor') and edge_app.system_coordinator.stream_processor:
            edge_app.system_coordinator.stream_processor.stop_processing()

        print(f"\n[RESULTS]")
        print(f"Frames received: {len(frames_received)}")
        if frames_received:
            print(f"Frame shapes: {set(frames_received)}")

        return len(frames_received) > 0

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_frame_callback():
    """Test just the frame callback mechanism."""
    print("\n=== Simple Frame Callback Test ===")

    frames_received = []

    def frame_callback(frame: np.ndarray):
        frames_received.append(frame.shape)
        print(f"CALLBACK RECEIVED: {frame.shape}")

    # Create a mock EdgeApplication-like object
    class MockEdgeApp:
        def __init__(self):
            self.frame_callback = frame_callback

    mock_app = MockEdgeApp()

    # Test the callback directly
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("1. Testing direct callback...")
    mock_app.frame_callback(test_frame)

    print(f"2. Frames received: {len(frames_received)}")
    return len(frames_received) > 0

if __name__ == "__main__":
    success1 = test_simple_frame_callback()
    success2 = test_detection_worker_isolation()

    print(f"\n[FINAL RESULT]")
    print(f"Frame callback test: {'SUCCESS' if success1 else 'FAILED'}")
    print(f"Detection worker test: {'SUCCESS' if success2 else 'FAILED'}")