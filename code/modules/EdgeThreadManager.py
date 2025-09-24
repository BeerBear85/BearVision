"""
Edge Thread Management Module

This module provides background thread management for the Edge Application system.
It handles BLE logging, video post-processing, and file uploading in separate threads
with proper coordination and error handling.
"""

import logging
import threading
import asyncio
import time
import queue
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from ble_beacon_handler import BleBeaconHandler
from GoogleDriveHandler import GoogleDriveHandler
from BoxHandler import BoxHandler
from FullClipExtractor import FullClipExtractor
from TrackerClipExtractor import TrackerClipExtractor


logger = logging.getLogger(__name__)


@dataclass
class VideoFile:
    """Represents a video file to be processed."""
    file_path: Path
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ProcessedVideo:
    """Represents a processed video ready for upload."""
    original_path: Path
    processed_path: Path
    metadata: Dict[str, Any]
    processing_time: float


class BLELoggerThread(threading.Thread):
    """Continuous BLE tag logging thread."""

    def __init__(self, status_callback: Optional[Callable[[str, Any], None]] = None):
        super().__init__(name="BLELogger", daemon=True)
        self.ble_handler = BleBeaconHandler()
        self.status_callback = status_callback
        self.running = False
        self.ble_data_queue = queue.Queue()

    def start_logging(self):
        """Start BLE logging."""
        self.running = True
        self.start()
        logger.info("BLE logging thread started")

    def stop_logging(self):
        """Stop BLE logging."""
        self.running = False
        logger.info("BLE logging thread stopped")

    def run(self):
        """Main BLE logging loop."""
        logger.info("BLE logging thread running")

        while self.running:
            try:
                # Start BLE scanning for a short duration
                self.ble_handler.start_scan(timeout=1.0)

                # Process any queued BLE data
                while not self.ble_data_queue.empty():
                    try:
                        ble_data = self.ble_data_queue.get_nowait()
                        if self.status_callback:
                            self.status_callback("ble_data", ble_data)
                        self.ble_data_queue.task_done()
                    except queue.Empty:
                        break

                time.sleep(0.1)  # Brief pause between scans

            except Exception as e:
                logger.error(f"BLE logging error: {e}")
                time.sleep(1.0)  # Longer pause on error


class PostProcessorThread(threading.Thread):
    """Video post-processing thread."""

    def __init__(self, video_queue: queue.Queue, processed_queue: queue.Queue,
                 status_callback: Optional[Callable[[str, Any], None]] = None):
        super().__init__(name="PostProcessor", daemon=True)
        self.video_queue = video_queue
        self.processed_queue = processed_queue
        self.status_callback = status_callback
        self.running = False

        # Initialize video processors
        self.full_clip_extractor = FullClipExtractor()
        self.tracker_clip_extractor = TrackerClipExtractor()

    def start_processing(self):
        """Start video processing."""
        self.running = True
        self.start()
        logger.info("Video post-processing thread started")

    def stop_processing(self):
        """Stop video processing."""
        self.running = False
        logger.info("Video post-processing thread stopped")

    def run(self):
        """Main video processing loop."""
        logger.info("Video post-processing thread running")

        while self.running:
            try:
                # Wait for video to process
                video_file = self.video_queue.get(timeout=1.0)

                if video_file is None:  # Shutdown signal
                    break

                logger.info(f"Processing video: {video_file.file_path}")

                if self.status_callback:
                    self.status_callback("processing_started", video_file)

                # Process the video file
                processed_video = self._process_video(video_file)

                if processed_video:
                    self.processed_queue.put(processed_video)
                    logger.info(f"Video processed successfully: {processed_video.processed_path}")

                    if self.status_callback:
                        self.status_callback("processing_completed", processed_video)
                else:
                    logger.error(f"Failed to process video: {video_file.file_path}")

                self.video_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Post-processing error: {e}")
                if not self.video_queue.empty():
                    self.video_queue.task_done()

    def _process_video(self, video_file: VideoFile) -> Optional[ProcessedVideo]:
        """Process a single video file."""
        try:
            start_time = time.time()

            # Determine processing type based on metadata
            use_tracker = video_file.metadata.get("use_tracker", False)

            if use_tracker:
                # Use tracker-based clip extraction
                processed_path = self._process_with_tracker(video_file)
            else:
                # Use full clip extraction
                processed_path = self._process_full_clip(video_file)

            if processed_path and processed_path.exists():
                processing_time = time.time() - start_time

                return ProcessedVideo(
                    original_path=video_file.file_path,
                    processed_path=processed_path,
                    metadata=video_file.metadata.copy(),
                    processing_time=processing_time
                )
            else:
                return None

        except Exception as e:
            logger.error(f"Error processing video {video_file.file_path}: {e}")
            return None

    def _process_with_tracker(self, video_file: VideoFile) -> Optional[Path]:
        """Process video using tracker clip extractor."""
        logger.info(f"Processing with tracker: {video_file.file_path}")

        output_dir = video_file.file_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"tracker_{video_file.file_path.name}"

        # TODO: Implement actual tracker processing
        # self.tracker_clip_extractor.extract_clips_from_list([clip_spec])

        return output_path

    def _process_full_clip(self, video_file: VideoFile) -> Optional[Path]:
        """Process video using full clip extractor."""
        logger.info(f"Processing full clip: {video_file.file_path}")

        output_dir = video_file.file_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"full_{video_file.file_path.name}"

        # TODO: Implement actual full clip processing
        # self.full_clip_extractor.extract_clips_from_list([clip_spec])

        return output_path


class UploaderThread(threading.Thread):
    """File upload thread for cloud storage."""

    def __init__(self, processed_queue: queue.Queue,
                 status_callback: Optional[Callable[[str, Any], None]] = None):
        super().__init__(name="Uploader", daemon=True)
        self.processed_queue = processed_queue
        self.status_callback = status_callback
        self.running = False

        # Initialize upload handlers
        try:
            self.google_drive_handler = GoogleDriveHandler()
            self.google_drive_available = True
        except Exception as e:
            logger.warning(f"Google Drive handler not available: {e}")
            self.google_drive_handler = None
            self.google_drive_available = False

        try:
            self.box_handler = BoxHandler()
            self.box_available = True
        except Exception as e:
            logger.warning(f"Box handler not available: {e}")
            self.box_handler = None
            self.box_available = False

    def start_uploading(self):
        """Start file uploading."""
        self.running = True
        self.start()
        logger.info("File upload thread started")

    def stop_uploading(self):
        """Stop file uploading."""
        self.running = False
        logger.info("File upload thread stopped")

    def run(self):
        """Main upload loop."""
        logger.info("File upload thread running")

        while self.running:
            try:
                # Wait for processed video to upload
                processed_video = self.processed_queue.get(timeout=1.0)

                if processed_video is None:  # Shutdown signal
                    break

                logger.info(f"Uploading video: {processed_video.processed_path}")

                if self.status_callback:
                    self.status_callback("upload_started", processed_video)

                # Upload the processed video
                success = self._upload_video(processed_video)

                if success:
                    logger.info(f"Video uploaded successfully: {processed_video.processed_path}")

                    if self.status_callback:
                        self.status_callback("upload_completed", processed_video)
                else:
                    logger.error(f"Failed to upload video: {processed_video.processed_path}")

                self.processed_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Upload error: {e}")
                if not self.processed_queue.empty():
                    self.processed_queue.task_done()

    def _upload_video(self, processed_video: ProcessedVideo) -> bool:
        """Upload a processed video to cloud storage."""
        try:
            # Try Google Drive first
            if self.google_drive_available and self.google_drive_handler:
                try:
                    self.google_drive_handler.upload_file(
                        str(processed_video.processed_path),
                        processed_video.processed_path.name
                    )
                    logger.info(f"Uploaded to Google Drive: {processed_video.processed_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Google Drive upload failed: {e}")

            # Try Box as fallback
            if self.box_available and self.box_handler:
                try:
                    self.box_handler.upload_file(
                        str(processed_video.processed_path),
                        processed_video.processed_path.name
                    )
                    logger.info(f"Uploaded to Box: {processed_video.processed_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Box upload failed: {e}")

            logger.error("No upload handlers available")
            return False

        except Exception as e:
            logger.error(f"Error uploading video {processed_video.processed_path}: {e}")
            return False


class EdgeThreadManager:
    """
    Manages all background threads for the Edge Application.

    This class coordinates:
    - BLE logging thread
    - Video post-processing thread
    - File upload thread
    - Inter-thread communication queues
    """

    def __init__(self, status_callback: Optional[Callable[[str, Any], None]] = None):
        """
        Initialize the Edge Thread Manager.

        Parameters
        ----------
        status_callback : Callable[[str, Any], None], optional
            Callback function for thread status updates
        """
        self.status_callback = status_callback

        # Inter-thread communication
        self.video_queue = queue.Queue()
        self.processed_queue = queue.Queue()

        # Background threads
        self.ble_thread: Optional[BLELoggerThread] = None
        self.processor_thread: Optional[PostProcessorThread] = None
        self.uploader_thread: Optional[UploaderThread] = None

        self.running = False

    def start_all_threads(self) -> bool:
        """Start all background threads."""
        try:
            # Initialize and start threads
            self.ble_thread = BLELoggerThread(status_callback=self.status_callback)
            self.processor_thread = PostProcessorThread(
                self.video_queue,
                self.processed_queue,
                status_callback=self.status_callback
            )
            self.uploader_thread = UploaderThread(
                self.processed_queue,
                status_callback=self.status_callback
            )

            # Start all threads
            self.ble_thread.start_logging()
            self.processor_thread.start_processing()
            self.uploader_thread.start_uploading()

            self.running = True
            logger.info("All background threads started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start background threads: {e}")
            return False

    def stop_all_threads(self) -> None:
        """Stop all background threads gracefully."""
        logger.info("Stopping all background threads...")
        self.running = False

        # Stop threads
        if self.ble_thread:
            self.ble_thread.stop_logging()

        if self.processor_thread:
            self.processor_thread.stop_processing()
            self.video_queue.put(None)  # Shutdown signal

        if self.uploader_thread:
            self.uploader_thread.stop_uploading()
            self.processed_queue.put(None)  # Shutdown signal

        # Wait for threads to finish
        threads = [self.ble_thread, self.processor_thread, self.uploader_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not shutdown gracefully")

        logger.info("All background threads stopped")

    def queue_video_for_processing(self, video_file: VideoFile) -> None:
        """Queue a video file for processing."""
        if self.running:
            self.video_queue.put(video_file)

    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes."""
        return {
            'video_queue': self.video_queue.qsize(),
            'processed_queue': self.processed_queue.qsize()
        }