"""
Stream Processing Module

This module handles video stream processing, frame capture, YOLO detection integration,
and frame callback management for the Edge Application system.
"""

import time
import threading
import cv2
import numpy as np
import logging
from typing import Optional, Callable, List, Dict, Any
from StatusManager import StatusManager, EdgeStatus, DetectionResult
from DnnHandler import DnnHandler


logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Handles video stream processing and frame detection.

    This class manages:
    - Video stream capture from GoPro cameras
    - Frame processing and YOLO detection integration
    - Detection box drawing and visualization
    - Frame callbacks to GUI components
    - Stream URL fallback and connection management
    """

    def __init__(self,
                 status_manager: StatusManager,
                 dnn_handler: Optional[DnnHandler] = None):
        """
        Initialize the Stream Processor.

        Parameters
        ----------
        status_manager : StatusManager
            Status manager for logging and status updates
        dnn_handler : DnnHandler, optional
            YOLO detection handler for person detection
        """
        self.status_manager = status_manager
        self.dnn_handler = dnn_handler

        # Stream processing state
        self.running = False
        self.detection_thread: Optional[threading.Thread] = None
        self.preview_stream_url: Optional[str] = None

        # Processing statistics
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_time = 0

        # Stream configuration
        self.detection_frame_interval = 5  # Run detection every 5th frame
        self.detection_confidence_threshold = 0.5

    def set_preview_stream_url(self, url: str) -> None:
        """Set the preview stream URL."""
        self.preview_stream_url = url
        self.status_manager.log("debug", f"Stream URL set to: {url}")

    def start_processing(self) -> bool:
        """
        Start frame processing in background thread.

        Returns
        -------
        bool
            True if processing started successfully, False otherwise
        """
        if self.running:
            self.status_manager.log("warning", "Stream processing already running")
            return True

        try:
            self.running = True
            self.detection_thread = threading.Thread(
                target=self._detection_worker,
                name="stream_processor",
                daemon=True
            )
            self.detection_thread.start()

            self.status_manager.log("info", "Stream processing started")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to start stream processing: {e}")
            self.running = False
            return False

    def stop_processing(self) -> None:
        """Stop frame processing."""
        self.status_manager.log("info", "Stopping stream processing...")
        self.running = False

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)
            if self.detection_thread.is_alive():
                self.status_manager.log("warning", "Stream processing thread did not stop gracefully")

        self.status_manager.log("info", "Stream processing stopped")

    def _get_stream_urls(self) -> List[str]:
        """Get list of stream URLs to try, with fallbacks."""
        urls_to_try = []

        # First, try the actual URL from GoPro controller if available
        if self.preview_stream_url:
            urls_to_try.append(self.preview_stream_url)
            self.status_manager.log("debug", f"Using GoPro controller URL: {self.preview_stream_url}")

        # Fallback to common GoPro IP addresses
        fallback_urls = [
            "udp://172.24.106.51:8554",  # Common wired IP
            "udp://172.20.110.51:8554",  # Alternative wired IP
            "udp://172.25.90.51:8554",   # Another alternative
            "udp://10.5.5.9:8554"       # WiFi IP
        ]

        # Add fallbacks (but avoid duplicates)
        for url in fallback_urls:
            if url not in urls_to_try:
                urls_to_try.append(url)

        return urls_to_try

    def _establish_video_capture(self) -> Optional[tuple]:
        """
        Establish video capture connection.

        Returns
        -------
        Optional[tuple]
            (capture_object, stream_url) if successful, None otherwise
        """
        urls_to_try = self._get_stream_urls()

        for url in urls_to_try:
            self.status_manager.log("debug", f"Trying preview stream URL: {url}")

            try:
                test_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                if test_cap.isOpened():
                    # Configure capture properties for better streaming
                    test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    test_cap.set(cv2.CAP_PROP_FPS, 30)

                    # Test if we can actually read frames
                    self.status_manager.log("debug", f"Testing frame read from {url}...")

                    successful_reads = 0
                    for i in range(3):
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None:
                            successful_reads += 1
                            self.status_manager.log("debug", f"Test read {i+1}: SUCCESS, frame shape: {test_frame.shape}")
                        else:
                            self.status_manager.log("debug", f"Test read {i+1}: FAILED, ret={ret}")
                        time.sleep(0.1)

                    if successful_reads >= 1:
                        self.status_manager.log("info", f"Successfully connected to preview stream: {url} ({successful_reads}/3 test reads)")

                        # Re-configure for optimal performance
                        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        test_cap.set(cv2.CAP_PROP_FPS, 30)

                        return test_cap, url
                    else:
                        self.status_manager.log("debug", f"Could not read frames from {url}")
                        test_cap.release()
                else:
                    self.status_manager.log("debug", f"Could not open {url}")
                    test_cap.release()

            except Exception as url_error:
                self.status_manager.log("debug", f"Error trying {url}: {url_error}")
                if 'test_cap' in locals():
                    test_cap.release()

        self.status_manager.log("warning", "Could not connect to any GoPro preview stream")
        return None

    def _detection_worker(self) -> None:
        """Main detection worker thread that processes preview frames."""
        cap = None
        stream_url = None
        no_frame_count = 0
        last_debug_time = time.time()

        # Establish video capture
        try:
            result = self._establish_video_capture()
            if result:
                cap, stream_url = result
            else:
                self.status_manager.log("error", "Failed to establish video capture")
                return

        except Exception as e:
            self.status_manager.log("error", f"Failed to initialize preview capture: {e}")
            return

        # Main frame processing loop
        self.status_manager.log("info", f"Starting frame processing loop with stream: {stream_url}")
        self.frame_count = 0

        while self.running and self.status_manager.status.preview_active:
            try:
                if cap and cap.isOpened():
                    # Try to read frame
                    ret, frame = cap.read()

                    if ret and frame is not None:
                        # Successfully read frame
                        self.frame_count += 1
                        no_frame_count = 0

                        # Create a copy for GUI to avoid threading issues
                        frame_copy = frame.copy()

                        # Run YOLO detection on every nth frame
                        detection_boxes = []
                        if self.frame_count % self.detection_frame_interval == 0:
                            detection_boxes = self._process_frame_detection_with_boxes(frame_copy)

                        # Draw detection boxes on the frame copy
                        if detection_boxes:
                            frame_copy = self._draw_detection_boxes(frame_copy, detection_boxes)

                        # Send frame to GUI via callback
                        self.status_manager.trigger_frame_callback(frame_copy)

                        # Log success periodically
                        if self.frame_count == 1 or self.frame_count % 30 == 0:
                            self.status_manager.log("debug", f"Processing frames successfully, count: {self.frame_count}")

                    else:
                        no_frame_count += 1

                        # Log frame reading issues periodically
                        current_time = time.time()
                        if current_time - last_debug_time > 5.0:
                            self.status_manager.log("warning", f"No frames from stream after {no_frame_count} attempts")
                            last_debug_time = current_time

                        # Brief pause before retrying
                        time.sleep(0.033)  # ~30fps attempt rate

                else:
                    # If no capture available
                    self.status_manager.log("error", f"Video capture not available")
                    time.sleep(1.0)
                    break

            except Exception as e:
                self.status_manager.log("error", f"Detection worker error: {e}")
                time.sleep(0.1)

        # Log why the loop ended and cleanup
        self.status_manager.log("info", f"Frame processing loop ended. Total frames processed: {self.frame_count}")

        if cap:
            cap.release()

    def _process_frame_detection_with_boxes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process frame for YOLO detection and return bounding boxes.

        Parameters
        ----------
        frame : np.ndarray
            Input frame for detection

        Returns
        -------
        List[Dict[str, Any]]
            List of detection boxes with coordinates, confidence, and labels
        """
        boxes = []

        if not self.dnn_handler:
            return boxes

        try:
            # Run YOLO detection on the frame
            detection_result = self.dnn_handler.find_person(frame)

            if detection_result and len(detection_result) == 2:
                detected_boxes, confidences = detection_result

                if detected_boxes and confidences:
                    for i, (box, confidence) in enumerate(zip(detected_boxes, confidences)):
                        if confidence > self.detection_confidence_threshold:
                            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                            x, y, w, h = box
                            x1, y1, x2, y2 = x, y, x + w, y + h

                            boxes.append({
                                'coords': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'label': 'Person'
                            })

            if boxes:
                # Update detection statistics
                self.detection_count += 1
                self.last_detection_time = time.time()

                # Notify status manager of detection
                detection = DetectionResult(
                    boxes=[[box['coords'][0], box['coords'][1],
                           box['coords'][2] - box['coords'][0],  # width
                           box['coords'][3] - box['coords'][1]]], # height
                    confidences=[box['confidence'] for box in boxes],
                    timestamp=self.last_detection_time
                )

                self.status_manager.trigger_detection_callback(detection)
                self.status_manager.update_status(overall_status=EdgeStatus.MOTION_DETECTED)
                self.status_manager.log("info", f"Person detected with {len(boxes)} bounding boxes!")

        except Exception as e:
            self.status_manager.log("error", f"Frame detection error: {e}")

        return boxes

    def _draw_detection_boxes(self, frame: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw YOLO detection boxes on frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame
        boxes : List[Dict[str, Any]]
            Detection boxes to draw

        Returns
        -------
        np.ndarray
            Frame with detection boxes drawn
        """
        try:
            for box in boxes:
                coords = box['coords']
                confidence = box['confidence']
                label = box['label']

                # Extract coordinates
                x1, y1, x2, y2 = map(int, coords)

                # Draw green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with confidence
                label_text = f"{label}: {confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), (0, 255, 0), -1)

                # Draw label text
                cv2.putText(frame, label_text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        except Exception as e:
            self.status_manager.log("error", f"Error drawing detection boxes: {e}")

        return frame

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'last_detection_time': self.last_detection_time,
            'is_running': self.running
        }