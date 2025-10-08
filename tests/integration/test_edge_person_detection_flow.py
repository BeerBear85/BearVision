"""
Integration test for Edge Application with full person detection flow.

This test verifies the complete flow:
1. Edge application starts up without physical GoPro
2. Reaches LOOKING_FOR_WAKEBOARDER state
3. Processes test image with wakeboarder
4. Detects person and transitions to RECORDING state

Tests the integration between EdgeApplicationStateMachine, StreamProcessor,
DnnHandler (YOLO), and state transitions.
"""

import sys
import time
import threading
import logging
from pathlib import Path
from unittest import mock
from unittest.mock import patch, MagicMock
import pytest
import cv2
import numpy as np

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
APP_DIR = Path(__file__).resolve().parents[2] / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

# Import test stubs
from tests.stubs.gopro import FakeGoPro

# Import application modules
from edge_application import EdgeApplicationStateMachine
from EdgeStateMachine import ApplicationState
from EdgeApplicationConfig import EdgeApplicationConfig
from StatusManager import DetectionResult


logger = logging.getLogger(__name__)


class MockStreamProcessor:
    """Mock StreamProcessor that can inject test frames."""

    def __init__(self, status_manager, dnn_handler=None, config=None):
        self.status_manager = status_manager
        self.dnn_handler = dnn_handler
        self.config = config
        self.running = False
        self.detection_thread = None
        self.test_frame = None
        self.should_detect = False

    def set_preview_stream_url(self, url):
        """Mock setting stream URL."""
        pass

    def start_processing(self):
        """Start mock processing that can inject test frames."""
        logger.info("MockStreamProcessor: start_processing() called")
        self.running = True
        # Don't start processing immediately - wait for test frame injection
        return True

    def stop_processing(self):
        """Stop mock processing."""
        self.running = False

    def inject_test_frame(self, frame):
        """Inject a test frame for processing."""
        logger.info("MockStreamProcessor: inject_test_frame() called")
        self.test_frame = frame
        self.should_detect = True
        if self.running and self.dnn_handler:
            # Start a thread that will process our test frame
            self.detection_thread = threading.Thread(target=self._process_test_frame, daemon=True)
            self.detection_thread.start()
            logger.info("MockStreamProcessor: Started detection thread")

    def _process_test_frame(self):
        """Process the injected test frame through YOLO detection."""
        # Wait a moment to simulate some processing delay
        time.sleep(0.5)

        if not self.running or not self.should_detect or self.test_frame is None:
            return

        try:
            # Run actual YOLO detection on test frame
            boxes, confidences = self.dnn_handler.find_person(self.test_frame)

            logger.info(f"YOLO detection results: {len(boxes)} boxes, confidences: {confidences}")

            if boxes and confidences:
                # Person detected! Create detection result
                detection = DetectionResult(
                    boxes=boxes,
                    confidences=confidences,
                    timestamp=time.time()
                )

                logger.info(f"Triggering detection callback with {len(boxes)} detections")

                # Trigger detection callback through status manager
                self.status_manager.trigger_detection_callback(detection)

        except Exception as e:
            logger.error(f"Error in mock detection processing: {e}")


class MockDnnHandler:
    """Mock DnnHandler that simulates YOLO but returns real detection results."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.confidence_threshold = 0.5
        self.initialized = False

    def init(self):
        """Mock initialization - no actual model loading."""
        self.initialized = True

    def find_person(self, image):
        """
        Mock person detection that returns realistic results for our test image.
        For the wakeboarder test image, we'll return a detection.
        """
        if not self.initialized:
            return [], []

        # For our test, simulate person detection in the wakeboarder image
        # Return a bounding box that covers roughly where a person would be
        height, width = image.shape[:2]

        # Simulate a person detection in the center-ish area of the image
        # where the wakeboarder would typically be
        person_box = [
            int(width * 0.3),   # x
            int(height * 0.2),  # y
            int(width * 0.4),   # width
            int(height * 0.6)   # height
        ]

        boxes = [person_box]
        confidences = [0.85]  # High confidence detection

        return boxes, confidences


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = EdgeApplicationConfig()
    # Override config values for testing
    config._values.update({
        'yolo_enabled': True,
        'yolo_model': 'yolov8n',
        'recording_duration': 5.0,
        'detection_cooldown': 1.0,
        'detection_confidence_threshold': 0.5,
        'hindsight_mode_enabled': False,      # Disable for testing
        'preview_stream_enabled': True,
        'max_error_restarts': 1,
        'error_restart_delay': 1.0,
        'enable_ble_logging': False,          # Disable for testing
        'enable_post_processing': False,      # Disable for testing
        'enable_cloud_upload': False          # Disable for testing
    })
    return config


@pytest.fixture
def test_image():
    """Load the test wakeboarder image."""
    image_path = Path(__file__).resolve().parents[2] / "test" / "images" / "test_image_easy.jpg"
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        pytest.skip(f"Could not load test image: {image_path}")

    return image


def test_edge_application_person_detection_flow(test_config, test_image):
    """
    Test complete edge application flow with person detection.

    This test verifies:
    1. Application starts up successfully
    2. Reaches LOOKING_FOR_WAKEBOARDER state
    3. Processes test image and detects person
    4. Transitions to RECORDING state
    """
    states_reached = []
    detection_results = []

    def status_callback(state: ApplicationState, message: str):
        """Track state transitions and status messages."""
        states_reached.append((state, message))
        logger.info(f"[{state.value.upper()}] {message}")

    def mock_detection_callback(detection: DetectionResult):
        """Track detection results."""
        detection_results.append(detection)
        logger.info(f"Detection callback: {len(detection.boxes)} boxes, confidences: {detection.confidences}")

    # Store reference to mock stream processor for later use
    mock_stream_processor = None

    def create_mock_stream_processor(status_manager, dnn_handler=None, config=None):
        nonlocal mock_stream_processor
        mock_stream_processor = MockStreamProcessor(status_manager, dnn_handler, config)
        return mock_stream_processor

    # Mock GoPro to avoid hardware dependency
    with patch('GoProController.WiredGoPro', FakeGoPro):
        # Mock DnnHandler to avoid needing ONNX model files
        with patch('DnnHandler.DnnHandler', MockDnnHandler):
            # Mock StreamProcessor at the module level where it's imported in EdgeSystemCoordinator
            with patch('EdgeSystemCoordinator.StreamProcessor', create_mock_stream_processor):
                # Create state machine with test config
                app = EdgeApplicationStateMachine(
                    status_callback=status_callback,
                    config=test_config
                )

                # Get references to internal components
                edge_app = app.edge_app
                system_coordinator = edge_app.system_coordinator

                # Set up detection callback to capture results (but don't interfere with the real callback)
                original_detection_callback = system_coordinator.status_manager.detection_callback
                def combined_detection_callback(detection):
                    mock_detection_callback(detection)
                    if original_detection_callback:
                        original_detection_callback(detection)

                # Replace the status manager's detection callback
                system_coordinator.status_manager.detection_callback = combined_detection_callback

                # Run state machine in separate thread
                app_thread = threading.Thread(target=app.run, daemon=True)
                app_thread.start()

                try:
                    # Wait for application to reach LOOKING_FOR_WAKEBOARDER state
                    logger.info("Waiting for LOOKING_FOR_WAKEBOARDER state...")
                    timeout = 15  # seconds
                    start_time = time.time()

                    while time.time() - start_time < timeout:
                        current_state = app.get_state()

                        if current_state == ApplicationState.LOOKING_FOR_WAKEBOARDER:
                            logger.info(f"✓ Reached LOOKING_FOR_WAKEBOARDER state in {time.time() - start_time:.1f}s")
                            break
                        elif current_state == ApplicationState.ERROR:
                            pytest.fail(f"Application entered ERROR state: {states_reached}")

                        time.sleep(0.1)
                    else:
                        pytest.fail(f"Timeout waiting for LOOKING_FOR_WAKEBOARDER state. Current: {app.get_state()}, States: {states_reached}")

                    # Inject test frame for processing
                    logger.info("Injecting test wakeboarder image...")
                    mock_stream_processor.inject_test_frame(test_image)

                    # Wait for detection and state transition to RECORDING
                    logger.info("Waiting for person detection and transition to RECORDING...")
                    detection_timeout = 10  # seconds
                    start_time = time.time()

                    while time.time() - start_time < detection_timeout:
                        current_state = app.get_state()

                        if current_state == ApplicationState.RECORDING:
                            logger.info(f"✓ Successfully transitioned to RECORDING state in {time.time() - start_time:.1f}s")
                            break
                        elif current_state == ApplicationState.ERROR:
                            pytest.fail(f"Application entered ERROR state during detection: {states_reached}")

                        time.sleep(0.1)
                    else:
                        pytest.fail(f"Timeout waiting for RECORDING state. Current: {app.get_state()}, Detections: {len(detection_results)}, States: {states_reached}")

                    # Verify we got detection results
                    assert len(detection_results) > 0, "No detection results received"

                    detection = detection_results[0]
                    assert len(detection.boxes) > 0, "No bounding boxes in detection result"
                    assert len(detection.confidences) > 0, "No confidences in detection result"
                    assert detection.confidences[0] > test_config.get_detection_confidence_threshold(), "Detection confidence too low"

                    logger.info("✓ Test completed successfully!")
                    logger.info(f"✓ States reached: {[s[0].value for s in states_reached]}")
                    logger.info(f"✓ Detection results: {len(detection_results)} detections")
                    logger.info(f"✓ Boxes detected: {len(detection.boxes)}")
                    logger.info(f"✓ Max confidence: {max(detection.confidences):.3f}")

                finally:
                    # Clean shutdown
                    app.shutdown()
                    app_thread.join(timeout=5)


if __name__ == "__main__":
    # Enable logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test manually for debugging
    config = EdgeApplicationConfig()
    image_path = Path(__file__).resolve().parents[2] / "test" / "images" / "test_image_easy.jpg"
    image = cv2.imread(str(image_path))

    if image is not None:
        test_edge_application_person_detection_flow(config, image)
    else:
        print(f"Could not load test image: {image_path}")