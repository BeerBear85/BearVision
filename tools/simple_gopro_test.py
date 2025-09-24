#!/usr/bin/env python3
"""
Simple GoPro connection test to isolate connection issues.
"""

import sys
import time
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
sys.path.append(str(MODULE_DIR))

from GoProController import GoProController

def test_basic_connection():
    """Test basic GoPro connection without advanced features."""
    print("=== Basic GoPro Connection Test ===")

    try:
        print("1. Creating GoProController...")
        gopro = GoProController()

        print("2. Attempting connection...")
        gopro.connect()
        print("   [OK] Connected successfully")

        print("3. Checking connection status...")
        print(f"   Connection status: {gopro._is_connected}")
        print(f"   GoPro object: {gopro._gopro}")

        if hasattr(gopro._gopro, '_serial'):
            print(f"   GoPro serial: {gopro._gopro._serial}")
        else:
            print("   [WARNING] No serial attribute")

        if hasattr(gopro._gopro, 'http_settings'):
            print("   [OK] http_settings available")
        else:
            print("   [WARNING] No http_settings attribute")

        print("4. Testing camera status...")
        try:
            status = gopro.get_camera_status()
            print(f"   [OK] Camera status retrieved: {len(str(status))} characters")
        except Exception as status_error:
            print(f"   [FAIL] Camera status error: {status_error}")

        print("5. Testing file list...")
        try:
            files = gopro.list_videos()
            print(f"   [OK] File list retrieved: {len(files)} files")
        except Exception as file_error:
            print(f"   [FAIL] File list error: {file_error}")

        print("6. Disconnecting...")
        gopro.disconnect()
        print("   [OK] Disconnected")

        return True

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

def test_manual_preview_start():
    """Test manual preview start without stream capture."""
    print("\n=== Manual Preview Start Test ===")

    try:
        print("1. Creating and connecting to GoPro...")
        gopro = GoProController()
        gopro.connect()

        print("2. Configuring GoPro...")
        gopro.configure()

        print("3. Starting preview manually...")
        preview_url = gopro.start_preview()
        print(f"   [OK] Preview URL: {preview_url}")

        print("4. Waiting 5 seconds...")
        time.sleep(5)

        print("5. Stopping preview...")
        gopro.stop_preview()

        print("6. Disconnecting...")
        gopro.disconnect()

        return preview_url

    except Exception as e:
        print(f"   [ERROR] {e}")
        return None

def main():
    """Main test function."""
    print("Simple GoPro Connection Test")
    print("=" * 40)

    # Test 1: Basic connection
    basic_success = test_basic_connection()

    # Test 2: Manual preview
    preview_url = test_manual_preview_start() if basic_success else None

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Basic connection: {'PASS' if basic_success else 'FAIL'}")
    print(f"Preview start: {'PASS' if preview_url else 'FAIL'}")

    if preview_url:
        print(f"Preview URL: {preview_url}")

    if not basic_success:
        print("\n[RECOMMENDATIONS]")
        print("1. Check that GoPro is connected via USB cable")
        print("2. Ensure GoPro is powered on and in wired mode")
        print("3. Check that no other applications are using the GoPro")
        print("4. Try disconnecting and reconnecting the GoPro")

if __name__ == "__main__":
    main()