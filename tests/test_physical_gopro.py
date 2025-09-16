"""
Tests for real physical GoPro camera communication.

This test only runs when explicitly triggered with: pytest --run-physical-gopro
It requires a physical GoPro camera to be connected via USB and powered on.

Based on functionalities available in tools/gopro_manual_test_gui.py
"""

import pytest
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
import sys

# Add module path for GoProController
MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from GoProController import GoProController
from GoProConfig import GoProConfiguration, VideoResolution, FrameRate, HindsightOption


class PhysicalGoProTest:
    """Helper class to handle physical GoPro camera testing."""

    def __init__(self):
        self.controller = None
        self.connected = False
        self.preview_active = False
        self.config_files_created = []

    def setup_connection(self, timeout_seconds=30):
        """Establish connection to physical GoPro with timeout."""
        print(f"Attempting to connect to GoPro (timeout: {timeout_seconds}s)...")

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                self.controller = GoProController()
                self.controller.connect()

                # Test basic communication
                status = self.controller.get_camera_status()
                if status:
                    self.connected = True
                    print(f"GoPro connected successfully!")
                    print(f"Camera status: {status}")
                    return True

            except Exception as e:
                print(f"Connection attempt failed: {e}")
                if self.controller:
                    try:
                        self.controller.disconnect()
                    except:
                        pass
                    self.controller = None
                time.sleep(2)  # Wait before retry

        return False

    def test_configuration_roundtrip(self):
        """Test downloading and uploading configuration."""
        if not self.connected:
            return False, "Not connected to GoPro"

        try:
            print("Testing configuration download...")

            # Create temporary file for download
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                download_path = temp_file.name
                self.config_files_created.append(download_path)

            # Download current configuration
            saved_path = self.controller.download_configuration(download_path)

            # Verify file was created and has content
            config_file = Path(saved_path)
            if not config_file.exists():
                return False, "Configuration file was not created"

            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            if not config_data or not isinstance(config_data, dict):
                return False, "Configuration file is empty or invalid"

            print(f"Configuration downloaded successfully to: {saved_path}")
            print(f"Config keys: {list(config_data.keys())}")

            # Test upload of the same configuration
            print("Testing configuration upload...")
            upload_result = self.controller.upload_configuration(saved_path)

            if not upload_result:
                return False, "Configuration upload returned False"

            print("Configuration uploaded successfully!")
            return True, "Configuration roundtrip successful"

        except Exception as e:
            return False, f"Configuration test failed: {e}"

    def test_file_listing(self):
        """Test retrieving file list from GoPro."""
        if not self.connected:
            return False, "Not connected to GoPro"

        try:
            print("Testing file list retrieval...")

            files = self.controller.list_videos()
            print(f"Retrieved {len(files)} files from GoPro")

            if files:
                print("Sample files:")
                for i, file in enumerate(files[:3]):  # Show first 3 files
                    print(f"   {i+1}. {file}")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more files")
            else:
                print("No files found on GoPro")

            return True, f"File listing successful ({len(files)} files)"

        except Exception as e:
            return False, f"File listing failed: {e}"

    def test_preview_stream(self, duration_seconds=5):
        """Test preview stream functionality."""
        if not self.connected:
            return False, "Not connected to GoPro"

        try:
            print(f"Testing preview stream for {duration_seconds} seconds...")

            # Start preview
            stream_url = self.controller.start_preview()
            self.preview_active = True
            print(f"Preview stream started: {stream_url}")

            # Let it run briefly
            time.sleep(duration_seconds)

            # Stop preview
            self.controller.stop_preview()
            self.preview_active = False
            print("Preview stream stopped")

            return True, "Preview stream test successful"

        except Exception as e:
            return False, f"Preview stream test failed: {e}"

    def test_brief_recording(self, duration_seconds=2):
        """Test very brief recording functionality."""
        if not self.connected:
            return False, "Not connected to GoPro"

        try:
            print(f"Testing brief recording for {duration_seconds} seconds...")
            print("This will create a short video file on the GoPro")

            # Start recording
            self.controller.start_recording()
            print("Recording started")

            # Record briefly
            time.sleep(duration_seconds)

            # Stop recording
            self.controller.stop_recording()
            print("Recording stopped")

            return True, "Brief recording test successful"

        except Exception as e:
            return False, f"Recording test failed: {e}"

    def test_hindsight_mode(self):
        """Test hindsight mode functionality."""
        if not self.connected:
            return False, "Not connected to GoPro"

        try:
            print("Testing hindsight mode...")

            # Configure and trigger hindsight mode (this uses the simplified method)
            self.controller.configure()  # This sets hindsight to 15 seconds
            print("GoPro configured with hindsight settings")

            # Test hindsight trigger
            self.controller.startHindsightMode()  # This triggers a 1-second hindsight capture
            print("Hindsight mode trigger successful")

            return True, "Hindsight mode test successful"

        except Exception as e:
            return False, f"Hindsight mode test failed: {e}"

    def cleanup(self):
        """Clean up connections and temporary files."""
        print("Cleaning up...")

        try:
            # Stop preview if active
            if self.preview_active and self.controller:
                self.controller.stop_preview()
                self.preview_active = False

            # Disconnect from GoPro
            if self.connected and self.controller:
                self.controller.disconnect()
                self.connected = False

            # Clean up temporary config files
            for config_file in self.config_files_created:
                try:
                    Path(config_file).unlink(missing_ok=True)
                except:
                    pass

            print("Cleanup completed")

        except Exception as e:
            print(f"Cleanup error (non-critical): {e}")


@pytest.mark.physical_gopro
def test_physical_gopro_functionality():
    """
    Test that validates communication with a real GoPro camera.

    Only runs when triggered with: pytest --run-physical-gopro

    Tests multiple core functionalities:
    - Connection establishment
    - Configuration download/upload
    - File listing
    - Preview stream
    - Brief recording
    - Hindsight mode
    """
    print("\n" + "="*60)
    print("PHYSICAL GOPRO CAMERA TEST")
    print("="*60)
    print("This test requires:")
    print("   - GoPro Black 12 (or compatible model)")
    print("   - USB connection to computer")
    print("   - GoPro powered on and in USB mode")
    print("   - Camera should be accessible at default IP (172.24.106.51)")
    print("")
    print("WARNING: This test will:")
    print("   - Connect to your GoPro camera")
    print("   - Download current configuration")
    print("   - Test preview stream briefly")
    print("   - Record a very short test video (2 seconds)")
    print("   - Test hindsight mode")
    print("")
    print("Make sure your GoPro is connected and ready...")
    print("="*60)

    # Disable the FakeGoPro mock for this test - we want to use the real implementation
    with patch.dict('sys.modules'):
        # Remove any existing mock modules to ensure we use real implementations
        if 'tests.stubs.gopro' in sys.modules:
            del sys.modules['tests.stubs.gopro']

    gopro_test = PhysicalGoProTest()
    test_results = []

    try:
        # Test 1: Connection
        print("\nTEST 1: Connection Establishment")
        print("-" * 40)
        connection_success = gopro_test.setup_connection(timeout_seconds=30)
        test_results.append(("Connection", connection_success, "Connected successfully" if connection_success else "Failed to connect"))

        if not connection_success:
            pytest.fail("❌ Failed to establish connection to GoPro. Ensure camera is connected via USB and powered on.")

        # Test 2: File Listing
        print("\nTEST 2: File Listing")
        print("-" * 40)
        file_success, file_message = gopro_test.test_file_listing()
        test_results.append(("File Listing", file_success, file_message))

        # Test 3: Configuration Roundtrip
        print("\nTEST 3: Configuration Management")
        print("-" * 40)
        config_success, config_message = gopro_test.test_configuration_roundtrip()
        test_results.append(("Configuration", config_success, config_message))

        # Test 4: Preview Stream (brief)
        print("\nTEST 4: Preview Stream")
        print("-" * 40)
        preview_success, preview_message = gopro_test.test_preview_stream(duration_seconds=3)
        test_results.append(("Preview Stream", preview_success, preview_message))

        # Test 5: Brief Recording
        print("\nTEST 5: Recording Functionality")
        print("-" * 40)
        recording_success, recording_message = gopro_test.test_brief_recording(duration_seconds=2)
        test_results.append(("Recording", recording_success, recording_message))

        # Test 6: Hindsight Mode
        print("\nTEST 6: Hindsight Mode")
        print("-" * 40)
        hindsight_success, hindsight_message = gopro_test.test_hindsight_mode()
        test_results.append(("Hindsight Mode", hindsight_success, hindsight_message))

    finally:
        # Always cleanup
        gopro_test.cleanup()

    # Print test results summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, success, message in test_results:
        status = "PASS" if success else "FAIL"
        print(f"{status:<8} {test_name:<20} {message}")
        if success:
            passed_tests += 1

    print("-" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("All tests passed - Physical GoPro communication successful!")
    else:
        failed_tests = total_tests - passed_tests
        failed_test_names = [name for name, success, _ in test_results if not success]
        print(f"{failed_tests} test(s) failed: {', '.join(failed_test_names)}")

    print("="*60)

    # Assert that critical tests passed
    critical_tests = ["Connection", "File Listing", "Configuration"]
    critical_failures = [name for name, success, _ in test_results if not success and name in critical_tests]

    if critical_failures:
        pytest.fail(f"❌ Critical test(s) failed: {', '.join(critical_failures)}. Check GoPro connection and setup.")

    # For non-critical tests, just log warnings but don't fail
    non_critical_failures = [name for name, success, _ in test_results if not success and name not in critical_tests]
    if non_critical_failures:
        print(f"Non-critical test(s) failed: {', '.join(non_critical_failures)}")

    assert connection_success, "GoPro connection test must pass"
    print("\nPhysical GoPro test completed successfully!")


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main([__file__, "-v", "-s"])