#!/usr/bin/env python3
"""
Direct UDP stream test to isolate video capture issues.
"""

import cv2
import time

def test_direct_udp_capture():
    """Test direct OpenCV capture from GoPro UDP stream."""
    print("=== Direct UDP Stream Test ===")

    url = "udp://172.24.106.51:8554"

    print(f"1. Opening video capture for: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    # Set some properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[FAIL] Could not open video capture")
        return False

    print("2. Video capture opened successfully")
    print(f"   Buffer size: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
    print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"   Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"   Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    print("3. Attempting to read frames...")
    frame_count = 0
    start_time = time.time()

    for attempt in range(100):  # Try 100 frame reads
        ret, frame = cap.read()

        if ret and frame is not None:
            frame_count += 1
            print(f"   [OK] Frame {frame_count}: {frame.shape}")

            if frame_count >= 5:  # Got 5 frames, that's success
                break
        else:
            print(f"   [FAIL] Attempt {attempt + 1}: ret={ret}, frame={'None' if frame is None else 'available'}")

        time.sleep(0.1)

        if time.time() - start_time > 10:  # 10 second timeout
            print("   [TIMEOUT] 10 seconds elapsed")
            break

    cap.release()

    elapsed = time.time() - start_time
    print(f"\n[RESULTS]")
    print(f"Frames successfully read: {frame_count}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Frame rate: {frame_count / elapsed:.2f} fps" if elapsed > 0 else "Frame rate: N/A")

    return frame_count > 0

def test_with_different_backends():
    """Test with different OpenCV backends."""
    print("\n=== Testing Different Backends ===")

    url = "udp://172.24.106.51:8554"
    backends = [
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY, "Any")
    ]

    for backend, name in backends:
        print(f"\nTesting with {name} backend...")
        try:
            cap = cv2.VideoCapture(url, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   [OK] {name}: Got frame {frame.shape}")
                    cap.release()
                    return True
                else:
                    print(f"   [FAIL] {name}: Could not read frame")
            else:
                print(f"   [FAIL] {name}: Could not open")
            cap.release()
        except Exception as e:
            print(f"   [ERROR] {name}: {e}")

    return False

if __name__ == "__main__":
    success1 = test_direct_udp_capture()
    success2 = test_with_different_backends()

    print(f"\n[FINAL RESULT]")
    print(f"Direct capture: {'SUCCESS' if success1 else 'FAILED'}")
    print(f"Backend test: {'SUCCESS' if success2 else 'FAILED'}")

    if not success1 and not success2:
        print("\n[TROUBLESHOOTING]")
        print("1. Ensure GoPro is streaming (run manual preview start test first)")
        print("2. Check if GoPro is actually transmitting UDP data")
        print("3. Try different network interfaces or IP addresses")
        print("4. Check firewall settings for UDP traffic")
        print("5. Verify OpenCV FFmpeg support is working")