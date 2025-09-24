#!/usr/bin/env python3
"""
Full integration test for EdgeApplication with actual GoPro connection.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
APP_DIR = Path(__file__).resolve().parents[1] / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

from edge_application import EdgeApplication, SystemStatus

def test_full_integration():
    """Test full EdgeApplication integration with GoPro and preview."""
    print("=== Full EdgeApplication Integration Test ===")

    frames_received = []
    log_messages = []

    def status_callback(status: SystemStatus):
        print(f"Status: {status.overall_status.value}, GoPro: {status.gopro_connected}, Preview: {status.preview_active}")

    def log_callback(level: str, message: str):
        log_messages.append((level, message))
        print(f"[{level.upper()}] {message}")

    def frame_callback(frame: np.ndarray):
        frames_received.append(time.time())
        print(f"FRAME #{len(frames_received)}: {frame.shape} at {time.time():.2f}")

    try:
        print("1. Creating EdgeApplication...")
        edge_app = EdgeApplication(
            status_callback=status_callback,
            log_callback=log_callback,
            frame_callback=frame_callback
        )

        print("2. Initializing system...")
        if not edge_app.initialize():
            print("[FAIL] System initialization failed")
            return False, 0

        print("3. Connecting to GoPro...")
        if not edge_app.connect_gopro():
            print("[FAIL] GoPro connection failed")
            return False, 0

        print("4. Starting preview...")
        if not edge_app.start_preview():
            print("[FAIL] Preview start failed")
            return False, 0

        print(f"5. Preview stream URL: {edge_app.preview_stream_url}")

        print("6. Starting detection processing...")
        if not edge_app._start_detection_processing():
            print("[FAIL] Detection processing start failed")
            return False, 0

        print("7. Waiting for frames (20 seconds)...")
        start_time = time.time()
        last_frame_count = 0

        while time.time() - start_time < 20:
            time.sleep(1)
            current_count = len(frames_received)

            if current_count > last_frame_count:
                print(f"   Progress: {current_count} frames received...")
                last_frame_count = current_count

            if current_count >= 5:  # Got some frames
                print("   Got enough frames, stopping early")
                break

        print("8. Stopping system...")
        edge_app.stop_system()

        print(f"\n[RESULTS]")
        print(f"Frames received: {len(frames_received)}")
        print(f"Log messages: {len(log_messages)}")

        if frames_received:
            frame_times = sorted(frames_received)
            durations = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
            avg_interval = sum(durations) / len(durations) if durations else 0
            fps = 1.0 / avg_interval if avg_interval > 0 else 0

            print(f"Average frame interval: {avg_interval:.3f}s")
            print(f"Estimated FPS: {fps:.2f}")

        # Show any error messages
        error_messages = [msg for level, msg in log_messages if level == 'error']
        warning_messages = [msg for level, msg in log_messages if level == 'warning']

        if error_messages:
            print(f"\nErrors ({len(error_messages)}):")
            for msg in error_messages[-3:]:  # Show last 3 errors
                print(f"  - {msg}")

        if warning_messages:
            print(f"\nWarnings ({len(warning_messages)}):")
            for msg in warning_messages[-3:]:  # Show last 3 warnings
                print(f"  - {msg}")

        return len(frames_received) > 0, len(frames_received)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False, 0

if __name__ == "__main__":
    success, frame_count = test_full_integration()

    print(f"\n{'='*50}")
    print(f"FINAL RESULT: {'SUCCESS' if success else 'FAILED'}")
    print(f"Total frames received: {frame_count}")

    if not success:
        print("\nNext steps:")
        if frame_count == 0:
            print("- Video capture is not working in the EdgeApplication context")
            print("- Check the detection worker thread implementation")
            print("- Verify video capture initialization in threaded environment")
        else:
            print("- Partial success, but may need optimization")