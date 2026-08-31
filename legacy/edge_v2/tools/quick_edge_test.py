#!/usr/bin/env python3
"""
Quick EdgeApplication test to verify preview integration.
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

def quick_test():
    """Quick test of EdgeApplication preview functionality."""
    print("=== Quick EdgeApplication Preview Test ===")

    frames_received = []
    status_updates = []

    def status_callback(status: SystemStatus):
        status_updates.append(status)
        print(f"Status: {status.overall_status.value}, GoPro: {status.gopro_connected}, Preview: {status.preview_active}")

    def log_callback(level: str, message: str):
        print(f"[{level.upper()}] {message}")

    def frame_callback(frame: np.ndarray):
        frames_received.append(frame.shape)
        print(f"FRAME RECEIVED: {frame.shape}")

    try:
        print("1. Initializing EdgeApplication...")
        edge_app = EdgeApplication(
            status_callback=status_callback,
            log_callback=log_callback,
            frame_callback=frame_callback
        )

        print("2. Initializing system...")
        if not edge_app.initialize():
            print("[FAIL] System initialization failed")
            return False

        print("3. Connecting to GoPro...")
        if not edge_app.connect_gopro():
            print("[FAIL] GoPro connection failed")
            return False

        print("4. Starting preview...")
        if not edge_app.start_preview():
            print("[FAIL] Preview start failed")
            return False

        print("5. Starting detection processing...")
        if not edge_app._start_detection_processing():
            print("[FAIL] Detection processing start failed")
            return False

        print("6. Waiting for frames (20 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 20:
            time.sleep(0.5)
            if len(frames_received) >= 3:  # Got some frames
                break

        print(f"\n[RESULTS]")
        print(f"Frames received: {len(frames_received)}")
        print(f"Status updates: {len(status_updates)}")
        print(f"Preview URL: {edge_app.preview_stream_url}")

        if frames_received:
            print(f"Frame shapes: {set(frames_received)}")

        print("7. Cleaning up...")
        edge_app.stop_system()

        return len(frames_received) > 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\n[FINAL RESULT] {'SUCCESS' if success else 'FAILED'}")