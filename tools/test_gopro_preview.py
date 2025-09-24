#!/usr/bin/env python3
"""
Test script for debugging GoPro connection and preview functionality.

This script helps identify issues with the preview pipeline in the EDGE GUI.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
APP_DIR = Path(__file__).resolve().parents[1] / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

from GoProController import GoProController
from edge_application import EdgeApplication, SystemStatus, DetectionResult

def test_gopro_controller_direct():
    """Test GoProController directly."""
    print("=== Testing GoProController directly ===")

    try:
        # Initialize GoPro controller
        print("1. Initializing GoProController...")
        gopro = GoProController()

        # Connect to GoPro
        print("2. Connecting to GoPro...")
        gopro.connect()
        print("   [OK] GoPro connected successfully")

        # Configure GoPro
        print("3. Configuring GoPro...")
        gopro.configure()
        print("   [OK] GoPro configured successfully")

        # Start preview
        print("4. Starting preview...")
        preview_url = gopro.start_preview()
        print(f"   [OK] Preview started, URL: {preview_url}")

        # Test video capture from preview URL
        print("5. Testing video capture from preview URL...")
        cap = cv2.VideoCapture(preview_url)

        if cap.isOpened():
            print("   [OK] Video capture opened successfully")

            # Try to read a few frames
            for i in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   [OK] Frame {i+1}: {frame.shape}")
                else:
                    print(f"   [FAIL] Failed to read frame {i+1}")
                time.sleep(0.1)

            cap.release()

        else:
            print(f"   [FAIL] Failed to open video capture for URL: {preview_url}")

        # Stop preview
        print("6. Stopping preview...")
        gopro.stop_preview()
        print("   [OK] Preview stopped")

        # Disconnect
        print("7. Disconnecting...")
        gopro.disconnect()
        print("   [OK] GoPro disconnected")

        return True

    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        return False

def test_edge_application():
    """Test EdgeApplication with callbacks."""
    print("\n=== Testing EdgeApplication ===")

    frames_received = []
    status_updates = []
    log_messages = []

    def status_callback(status: SystemStatus):
        status_updates.append(status)
        print(f"   Status update: {status.overall_status.value}, GoPro connected: {status.gopro_connected}, Preview active: {status.preview_active}")

    def detection_callback(detection: DetectionResult):
        print(f"   Detection: {len(detection.boxes)} boxes, confidences: {detection.confidences}")

    def log_callback(level: str, message: str):
        log_messages.append((level, message))
        print(f"   [{level.upper()}] {message}")

    def frame_callback(frame: np.ndarray):
        frames_received.append(frame.shape)
        print(f"   Frame received: {frame.shape}")

    try:
        # Initialize EdgeApplication
        print("1. Initializing EdgeApplication...")
        edge_app = EdgeApplication(
            status_callback=status_callback,
            detection_callback=detection_callback,
            log_callback=log_callback,
            frame_callback=frame_callback
        )

        # Initialize the system
        print("2. Initializing system...")
        if not edge_app.initialize():
            print("   [FAIL] Failed to initialize EdgeApplication")
            return False
        print("   [OK] EdgeApplication initialized")

        # Connect to GoPro
        print("3. Connecting to GoPro...")
        if not edge_app.connect_gopro():
            print("   [FAIL] Failed to connect to GoPro")
            return False
        print("   [OK] GoPro connected through EdgeApplication")

        # Start preview
        print("4. Starting preview...")
        if not edge_app.start_preview():
            print("   [FAIL] Failed to start preview")
            return False
        print("   [OK] Preview started through EdgeApplication")

        # Wait for frames
        print("5. Waiting for frames (10 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 10:
            time.sleep(0.1)
            if len(frames_received) > 0:
                break

        if frames_received:
            print(f"   [OK] Received {len(frames_received)} frames")
            print(f"   [OK] Frame shapes: {set(frames_received)}")
        else:
            print("   [FAIL] No frames received")

        # Stop preview
        print("6. Stopping preview...")
        edge_app.stop_preview()
        print("   [OK] Preview stopped")

        # Stop system
        print("7. Stopping system...")
        edge_app.stop_system()
        print("   [OK] System stopped")

        # Summary
        print(f"\n   Summary:")
        print(f"   - Status updates: {len(status_updates)}")
        print(f"   - Log messages: {len(log_messages)}")
        print(f"   - Frames received: {len(frames_received)}")

        return len(frames_received) > 0

    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        return False

def test_opencv_udp_direct():
    """Test OpenCV UDP capture directly with common GoPro IP."""
    print("\n=== Testing OpenCV UDP capture directly ===")

    # Common GoPro wired IP addresses to try
    test_urls = [
        "udp://172.24.106.51:8554",
        "udp://172.20.110.51:8554",
        "udp://172.25.90.51:8554",
        "udp://10.5.5.9:8554"  # GoPro WiFi IP
    ]

    for url in test_urls:
        print(f"Testing URL: {url}")

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame

        if cap.isOpened():
            print(f"   [OK] Opened successfully")

            # Try to read frames with timeout
            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < 5:  # 5 second timeout
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_count += 1
                    print(f"   [OK] Frame {frame_count}: {frame.shape}")
                    if frame_count >= 3:  # Got some frames, success
                        break
                time.sleep(0.1)

            cap.release()

            if frame_count > 0:
                print(f"   [OK] Successfully received {frame_count} frames from {url}")
                return url
            else:
                print(f"   [FAIL] No frames received from {url}")
        else:
            print(f"   [FAIL] Failed to open {url}")

    print("   [FAIL] No working UDP stream found")
    return None

def main():
    """Main test function."""
    print("GoPro Preview Debug Test")
    print("=" * 50)

    # Test 1: Direct GoProController test
    direct_success = test_gopro_controller_direct()

    # Test 2: EdgeApplication test
    edge_success = test_edge_application()

    # Test 3: Direct OpenCV UDP test
    working_url = test_opencv_udp_direct()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"GoProController direct test: {'[OK] PASS' if direct_success else '[FAIL] FAIL'}")
    print(f"EdgeApplication test: {'[OK] PASS' if edge_success else '[FAIL] FAIL'}")
    print(f"OpenCV UDP direct test: {'[OK] PASS' if working_url else '[FAIL] FAIL'}")

    if working_url:
        print(f"Working UDP URL: {working_url}")

    if not any([direct_success, edge_success, working_url]):
        print("\n[WARNING]  All tests failed. Possible issues:")
        print("   1. GoPro is not connected or not in wired mode")
        print("   2. GoPro IP address is different than expected")
        print("   3. Preview stream is not working on the GoPro")
        print("   4. Network connectivity issues")
        print("   5. OpenCV/FFmpeg UDP support issues")
    elif working_url and not edge_success:
        print(f"\n[WARNING]  Direct UDP works but EdgeApplication fails")
        print("   The issue is likely in the EdgeApplication preview integration")
        print("   Need to fix the stream URL handling in _detection_worker")

if __name__ == "__main__":
    main()