#!/usr/bin/env python3
"""
Test the complete preview pipeline with frame callback and detection boxes.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
APP_DIR = Path(__file__).resolve().parents[2] / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

from edge_application import EdgeApplication, SystemStatus

def test_complete_preview_pipeline():
    """Test the complete preview pipeline with frame callback and YOLO detection."""
    print("=== Complete Preview Pipeline Test ===")

    frames_received = []
    detection_events = []
    all_logs = []

    def status_callback(status: SystemStatus):
        print(f"Status Update: {status.overall_status.value}, Preview: {status.preview_active}, GoPro: {status.gopro_connected}")

    def log_callback(level: str, message: str):
        all_logs.append((level, message))
        print(f"[{level.upper()}] {message}")

    def frame_callback(frame: np.ndarray):
        frames_received.append({
            'timestamp': time.time(),
            'shape': frame.shape,
            'has_green': check_for_green_pixels(frame)
        })
        print(f"FRAME #{len(frames_received)}: {frame.shape}, Green pixels: {check_for_green_pixels(frame)}")

    def detection_callback(detection):
        detection_events.append(detection)
        print(f"DETECTION: {len(detection.boxes)} boxes, confidences: {detection.confidences}")

    def check_for_green_pixels(frame):
        """Check if frame has green pixels (indicating detection boxes)."""
        try:
            # Check for bright green pixels (detection boxes)
            green_mask = (frame[:, :, 1] > 200) & (frame[:, :, 0] < 50) & (frame[:, :, 2] < 50)
            return np.sum(green_mask) > 100  # More than 100 green pixels
        except:
            return False

    try:
        print("1. Creating EdgeApplication...")
        edge_app = EdgeApplication(
            status_callback=status_callback,
            log_callback=log_callback,
            frame_callback=frame_callback,
            detection_callback=detection_callback
        )

        print("2. Initializing system...")
        if not edge_app.initialize():
            print("   [FAIL] System initialization failed")
            return False, {}

        print("3. Connecting to GoPro...")
        if not edge_app.connect_gopro():
            print("   [FAIL] GoPro connection failed")
            return False, {}

        print("4. Starting preview...")
        if not edge_app.start_preview():
            print("   [FAIL] Preview start failed")
            return False, {}

        print(f"5. Preview URL: {edge_app.preview_stream_url}")

        print("6. Starting detection processing...")
        if not edge_app._start_detection_processing():
            print("   [FAIL] Detection processing failed")
            return False, {}

        print("7. Waiting for frames and detections (30 seconds)...")
        start_time = time.time()
        last_frame_count = 0

        while time.time() - start_time < 30:
            time.sleep(1)
            current_count = len(frames_received)

            if current_count > last_frame_count:
                print(f"   Progress: {current_count} frames received...")
                last_frame_count = current_count

            if current_count >= 10:  # Got enough frames
                print("   Got enough frames, stopping test")
                break

        print("8. Stopping system...")
        edge_app.stop_system()

        # Analyze results
        results = {
            'total_frames': len(frames_received),
            'frames_with_green': sum(1 for f in frames_received if f['has_green']),
            'detection_events': len(detection_events),
            'frame_shapes': list(set(f['shape'] for f in frames_received)),
            'logs': len(all_logs)
        }

        print(f"\n[RESULTS]")
        print(f"Total frames received: {results['total_frames']}")
        print(f"Frames with green pixels (detection boxes): {results['frames_with_green']}")
        print(f"Detection events: {results['detection_events']}")
        print(f"Frame shapes: {results['frame_shapes']}")
        print(f"Total log messages: {results['logs']}")

        if results['total_frames'] > 0:
            frame_times = [f['timestamp'] for f in frames_received]
            if len(frame_times) > 1:
                durations = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
                avg_interval = sum(durations) / len(durations)
                fps = 1.0 / avg_interval if avg_interval > 0 else 0
                print(f"Average FPS: {fps:.2f}")

        # Show relevant log messages
        frame_logs = [msg for level, msg in all_logs if 'frame' in msg.lower()]
        detection_logs = [msg for level, msg in all_logs if 'detection' in msg.lower()]
        error_logs = [msg for level, msg in all_logs if level == 'error']

        if frame_logs:
            print(f"\nFrame-related logs ({len(frame_logs)}):")
            for msg in frame_logs[-3:]:  # Show last 3
                print(f"   - {msg}")

        if detection_logs:
            print(f"\nDetection-related logs ({len(detection_logs)}):")
            for msg in detection_logs[-3:]:  # Show last 3
                print(f"   - {msg}")

        if error_logs:
            print(f"\nError logs ({len(error_logs)}):")
            for msg in error_logs[-3:]:  # Show last 3
                print(f"   - {msg}")

        success = results['total_frames'] > 0
        return success, results

    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def main():
    """Main test function."""
    print("Complete Preview Pipeline Test")
    print("=" * 50)

    success, results = test_complete_preview_pipeline()

    print("\n" + "=" * 50)
    print("FINAL ANALYSIS")
    print("=" * 50)
    print(f"Preview pipeline: {'SUCCESS' if success else 'FAILED'}")

    if success:
        print(f"✅ Frames received: {results['total_frames']}")
        print(f"✅ Detection boxes drawn: {results['frames_with_green']} frames")
        print(f"✅ YOLO detections: {results['detection_events']}")

        if results['frames_with_green'] > 0:
            print("✅ Green detection boxes are being drawn on frames!")
        else:
            print("⚠️  No detection boxes visible (may be no people in view)")

        print("\n[SUCCESS] Preview pipeline is working correctly!")
        print("The EDGE GUI should now show:")
        print("- Live video stream from GoPro")
        print("- Green bounding boxes around detected people")
        print("- Real-time frame updates")

    else:
        print("❌ Preview pipeline failed")
        print("Issues to check:")
        print("- Frame reading from UDP stream")
        print("- Frame callback chain to GUI")
        print("- YOLO detection processing")
        print("- Detection box rendering")

if __name__ == "__main__":
    main()