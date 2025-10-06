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
import queue
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
                 dnn_handler: Optional[DnnHandler] = None,
                 config: Optional['EdgeApplicationConfig'] = None):
        """
        Initialize the Stream Processor.

        Parameters
        ----------
        status_manager : StatusManager
            Status manager for logging and status updates
        dnn_handler : DnnHandler, optional
            YOLO detection handler for person detection
        config : EdgeApplicationConfig, optional
            Configuration object with stream settings
        """
        self.status_manager = status_manager
        self.dnn_handler = dnn_handler
        self.config = config

        # Stream processing state
        self.running = False
        self.detection_thread: Optional[threading.Thread] = None
        self.frame_callback_thread: Optional[threading.Thread] = None
        self.preview_stream_url: Optional[str] = None

        # Processing statistics
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_time = 0

        # Stream performance metrics
        self.frames_dropped = 0
        self.frames_processed = 0
        self.current_fps = 0.0
        self.current_lag_ms = 0.0
        self.last_frame_time = 0.0
        self.fps_update_time = time.time()
        self.fps_frame_count = 0

        # Stream configuration (from config or defaults)
        self.detection_frame_interval = 5  # Run detection every 5th frame

        if self.config:
            self.detection_confidence_threshold = self.config.get_detection_confidence_threshold()
            self.max_fps = self.config.get_stream_max_fps()
            self.max_lag_ms = self.config.get_stream_max_lag_ms()
            self.buffer_drain_enabled = self.config.get_stream_buffer_drain()
            callback_queue_size = self.config.get_stream_callback_queue_size()
        else:
            # Default values
            self.detection_confidence_threshold = 0.5
            self.max_fps = 30
            self.max_lag_ms = 500
            self.buffer_drain_enabled = True
            callback_queue_size = 2

        # Frame callback queue (non-blocking)
        self.frame_callback_queue = queue.Queue(maxsize=callback_queue_size)
        self.callback_running = False

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

            # Start frame callback worker thread
            self.frame_callback_thread = threading.Thread(
                target=self._frame_callback_worker,
                name="frame_callback",
                daemon=True
            )
            self.frame_callback_thread.start()

            # Start detection worker thread
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
        self.callback_running = False

        # Stop frame callback thread
        if self.frame_callback_thread and self.frame_callback_thread.is_alive():
            self.frame_callback_queue.put(None)  # Shutdown signal
            self.frame_callback_thread.join(timeout=2.0)
            if self.frame_callback_thread.is_alive():
                self.status_manager.log("warning", "Frame callback thread did not stop gracefully")

        # Stop detection thread
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

    def _get_latest_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """
        Get the latest frame from capture, draining buffer of old frames.

        This method reads multiple frames to drain OpenCV's internal buffer,
        ensuring we get the most recent frame and minimize lag.

        Parameters
        ----------
        cap : cv2.VideoCapture
            Video capture object

        Returns
        -------
        Optional[np.ndarray]
            Latest frame if available, None otherwise
        """
        if not self.buffer_drain_enabled:
            # Simple read without draining
            ret, frame = cap.read()
            return frame if ret else None

        # Drain buffer by reading multiple times
        frame = None
        frames_read = 0
        max_drain = 5  # Maximum frames to drain in one call

        for _ in range(max_drain):
            ret, temp_frame = cap.read()
            if ret and temp_frame is not None:
                frame = temp_frame
                frames_read += 1
            else:
                break

        # Track dropped frames (frames we skipped to get latest)
        if frames_read > 1:
            self.frames_dropped += (frames_read - 1)

        return frame

    def _frame_callback_worker(self) -> None:
        """
        Worker thread for non-blocking frame callbacks.

        This thread processes frames from the callback queue and sends them
        to the GUI without blocking the main detection loop.
        """
        self.callback_running = True

        while self.callback_running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_callback_queue.get(timeout=0.5)

                # Check for shutdown signal
                if frame is None:
                    break

                # Send frame to GUI via status manager
                self.status_manager.trigger_frame_callback(frame)
                self.frame_callback_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.status_manager.log("error", f"Frame callback worker error: {e}")

        self.status_manager.log("debug", "Frame callback worker stopped")

    def _send_frame_to_callback(self, frame: np.ndarray) -> None:
        """
        Send frame to callback queue (non-blocking).

        If queue is full, drop oldest frame and add new one.

        Parameters
        ----------
        frame : np.ndarray
            Frame to send to callback
        """
        try:
            # Try to put frame in queue without blocking
            self.frame_callback_queue.put_nowait(frame)
        except queue.Full:
            # Queue is full, drop oldest frame and add new one
            try:
                self.frame_callback_queue.get_nowait()  # Remove old frame
                self.frame_callback_queue.put_nowait(frame)  # Add new frame
                self.frames_dropped += 1
            except:
                pass

    def _update_fps_metrics(self) -> None:
        """Update FPS calculation."""
        current_time = time.time()
        self.fps_frame_count += 1

        # Update FPS every second
        time_diff = current_time - self.fps_update_time
        if time_diff >= 1.0:
            self.current_fps = self.fps_frame_count / time_diff
            self.fps_frame_count = 0
            self.fps_update_time = current_time

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
        frame_start_time = time.time()

        while self.running and self.status_manager.status.preview_active:
            try:
                loop_start = time.time()

                if cap and cap.isOpened():
                    # Get latest frame (draining buffer if enabled)
                    frame = self._get_latest_frame(cap)

                    if frame is not None:
                        # Successfully read frame
                        self.frame_count += 1
                        self.frames_processed += 1
                        no_frame_count = 0

                        # Calculate current lag
                        current_time = time.time()
                        if self.last_frame_time > 0:
                            frame_delta = current_time - self.last_frame_time
                            expected_delta = 1.0 / self.max_fps
                            self.current_lag_ms = max(0, (frame_delta - expected_delta) * 1000)
                        self.last_frame_time = current_time

                        # Update FPS metrics
                        self._update_fps_metrics()

                        # Check if we should drop this frame due to lag
                        should_process = True
                        if self.current_lag_ms > self.max_lag_ms:
                            self.frames_dropped += 1
                            should_process = False
                            if self.frame_count % 30 == 0:
                                self.status_manager.log("warning",
                                    f"Dropping frames due to lag ({self.current_lag_ms:.0f}ms)")

                        if should_process:
                            # Create a copy for processing to avoid threading issues
                            frame_copy = frame.copy()

                            # Run YOLO detection on every nth frame
                            detection_boxes = []
                            if self.frame_count % self.detection_frame_interval == 0:
                                detection_boxes = self._process_frame_detection_with_boxes(frame_copy)

                            # Draw detection boxes on the frame copy
                            if detection_boxes:
                                frame_copy = self._draw_detection_boxes(frame_copy, detection_boxes)

                            # Send frame to GUI via non-blocking callback
                            self._send_frame_to_callback(frame_copy)

                        # Log success periodically
                        if self.frame_count == 1 or self.frame_count % 60 == 0:
                            self.status_manager.log("debug",
                                f"Stream stats: {self.current_fps:.1f} FPS, "
                                f"{self.frames_dropped} dropped, "
                                f"lag: {self.current_lag_ms:.0f}ms")

                    else:
                        no_frame_count += 1

                        # Log frame reading issues periodically
                        current_time = time.time()
                        if current_time - last_debug_time > 5.0:
                            self.status_manager.log("warning", f"No frames from stream after {no_frame_count} attempts")
                            last_debug_time = current_time

                        # Brief pause before retrying
                        time.sleep(0.033)  # ~30fps attempt rate

                    # Adaptive frame rate control - sleep to maintain target FPS
                    loop_time = time.time() - loop_start
                    target_frame_time = 1.0 / self.max_fps
                    sleep_time = target_frame_time - loop_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)

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
                # Convert all boxes from [x1, y1, x2, y2] back to [x, y, w, h] for DetectionResult
                detection = DetectionResult(
                    boxes=[[b['coords'][0], b['coords'][1],
                            b['coords'][2] - b['coords'][0],  # width
                            b['coords'][3] - b['coords'][1]] for b in boxes],  # height
                    confidences=[b['confidence'] for b in boxes],
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
            'is_running': self.running,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'current_fps': self.current_fps,
            'current_lag_ms': self.current_lag_ms,
            'callback_queue_size': self.frame_callback_queue.qsize()
        }

    def get_stream_stats(self) -> Dict[str, Any]:
        """
        Get detailed stream performance statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing stream performance metrics
        """
        return {
            'fps': round(self.current_fps, 2),
            'lag_ms': round(self.current_lag_ms, 1),
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'drop_rate_percent': round(
                (self.frames_dropped / max(self.frames_processed, 1)) * 100, 1
            ) if self.frames_processed > 0 else 0,
            'callback_queue_size': self.frame_callback_queue.qsize(),
            'is_running': self.running,
            'buffer_drain_enabled': self.buffer_drain_enabled
        }