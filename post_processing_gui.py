#!/usr/bin/env python3
"""
Post-Processing Pipeline Testing GUI

A PySide6-based GUI for manually running and inspecting the post-processing pipeline
on a single video. Provides video playback with bounding box overlay visualization,
pipeline execution, and overlay image export functionality.

Features:
- Input video picker (default: regression test clip)
- Output directory picker
- Video playback with detection/trajectory overlay
- Play/pause/seek controls
- Run post-processing pipeline on demand
- Export overlay image (all boxes on last frame)
- Status display and logging

Usage:
    python post_processing_gui.py

Dependencies:
    - PySide6 (GUI framework)
    - OpenCV (video processing)
    - PostProcessingPipeline (core pipeline)

GitHub Issue: #167
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QSlider,
    QGroupBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap

# Add code/modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "modules"))
from PostProcessingPipeline import PostProcessingPipeline
from PostProcessingConfig import PostProcessingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("post_processing_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Default paths
DEFAULT_VIDEO = os.path.join("test", "input_video", "TestMovie3.avi")
DEFAULT_OUTPUT_DIR = os.path.join("output", "post_processing_gui")


class PipelineWorker(QThread):
    """Background worker thread for running the post-processing pipeline."""

    finished = Signal(dict)  # Pipeline results
    error = Signal(str)  # Error message
    progress = Signal(str)  # Progress updates

    def __init__(self, input_video: str, output_dir: str):
        super().__init__()
        self.input_video = input_video
        self.output_dir = output_dir

    def run(self):
        try:
            # Create output directory if needed
            os.makedirs(self.output_dir, exist_ok=True)

            # Generate output paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_json = os.path.join(self.output_dir, f"metadata_{timestamp}.json")
            output_video = os.path.join(self.output_dir, f"cropped_{timestamp}.mp4")

            self.progress.emit(f"Creating pipeline configuration...")

            # Create configuration
            config = PostProcessingConfig(
                input_video=self.input_video,
                output_json=output_json,
                output_video=output_video,
                yolo_model="yolov8n.pt",
                confidence_threshold=0.5,
                scaling_factor=1.5,
                cutoff_hz=2.0,
                verbose=True
            )

            self.progress.emit(f"Running pipeline on {os.path.basename(self.input_video)}...")

            # Create and run pipeline
            pipeline = PostProcessingPipeline(config)
            results = pipeline.run()

            self.progress.emit("Pipeline completed successfully")
            self.finished.emit(results)

        except Exception as e:
            logger.exception(f"Pipeline worker error: {e}")
            self.error.emit(str(e))


class PostProcessingGUI(QMainWindow):
    """Main window for post-processing pipeline testing GUI."""

    def __init__(self):
        super().__init__()

        # State variables
        self.input_video_path: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.pipeline_results: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.worker: Optional[PipelineWorker] = None

        # Video playback state
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_idx: int = 0
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.is_playing: bool = False
        self.playback_timer: QTimer = QTimer()
        self.playback_timer.timeout.connect(self.update_frame)

        self.init_ui()
        self.set_default_paths()
        logger.info("Post-Processing GUI initialized")

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Post-Processing Pipeline Testing Tool")
        self.setMinimumSize(1000, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)

        # Title
        title = QLabel("Post-Processing Pipeline Testing Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Input section
        input_group = self.create_input_section()
        main_layout.addWidget(input_group)

        # Video player section
        player_group = self.create_player_section()
        main_layout.addWidget(player_group, stretch=1)

        # Action buttons
        action_layout = self.create_action_section()
        main_layout.addLayout(action_layout)

        # Status section
        status_group = self.create_status_section()
        main_layout.addWidget(status_group)

        central_widget.setLayout(main_layout)

    def create_input_section(self) -> QGroupBox:
        """Create input file selection section."""
        group = QGroupBox("Input Configuration")
        layout = QVBoxLayout()

        # Video file selection
        video_layout = QHBoxLayout()
        video_label = QLabel("Input Video:")
        video_label.setMinimumWidth(100)
        video_layout.addWidget(video_label)

        self.video_path_label = QLabel("No file selected")
        self.video_path_label.setStyleSheet(
            "padding: 5px; border: 1px solid #ccc; background: #f9f9f9;"
        )
        video_layout.addWidget(self.video_path_label, stretch=1)

        self.browse_video_btn = QPushButton("Browse...")
        self.browse_video_btn.clicked.connect(self.browse_video)
        video_layout.addWidget(self.browse_video_btn)

        layout.addLayout(video_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Dir:")
        output_label.setMinimumWidth(100)
        output_layout.addWidget(output_label)

        self.output_dir_label = QLabel("No directory selected")
        self.output_dir_label.setStyleSheet(
            "padding: 5px; border: 1px solid #ccc; background: #f9f9f9;"
        )
        output_layout.addWidget(self.output_dir_label, stretch=1)

        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.browse_output_btn)

        layout.addLayout(output_layout)

        group.setLayout(layout)
        return group

    def create_player_section(self) -> QGroupBox:
        """Create video player section with controls."""
        group = QGroupBox("Video Player")
        layout = QVBoxLayout()

        # Video display area
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #000; color: #fff; border: 2px solid #333;"
        )
        layout.addWidget(self.video_label)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        controls_layout.addWidget(self.frame_slider, stretch=1)

        # Frame counter
        self.frame_counter_label = QLabel("Frame: 0 / 0")
        controls_layout.addWidget(self.frame_counter_label)

        layout.addLayout(controls_layout)

        group.setLayout(layout)
        return group

    def create_action_section(self) -> QHBoxLayout:
        """Create action buttons section."""
        layout = QHBoxLayout()

        self.run_pipeline_btn = QPushButton("Run Pipeline")
        self.run_pipeline_btn.setToolTip("Execute post-processing pipeline on selected video")
        self.run_pipeline_btn.clicked.connect(self.run_pipeline)
        self.run_pipeline_btn.setEnabled(False)
        self.run_pipeline_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.run_pipeline_btn)

        self.export_overlay_btn = QPushButton("Export Overlay Image")
        self.export_overlay_btn.setToolTip("Export last frame with all bounding boxes overlaid")
        self.export_overlay_btn.clicked.connect(self.export_overlay_image)
        self.export_overlay_btn.setEnabled(False)
        self.export_overlay_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.export_overlay_btn)

        return layout

    def create_status_section(self) -> QGroupBox:
        """Create status display section."""
        group = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.status_text)

        info_label = QLabel("Note: All operations are logged to post_processing_gui.log")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)

        group.setLayout(layout)
        return group

    def update_status(self, message: str, level: str = "info"):
        """Update status display with message."""
        if level == "error":
            logger.error(message)
            prefix = "ERROR: "
        elif level == "warning":
            logger.warning(message)
            prefix = "WARNING: "
        else:
            logger.info(message)
            prefix = ""

        self.status_text.append(f"{prefix}{message}")
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.End)
        self.status_text.setTextCursor(cursor)

    def set_default_paths(self):
        """Set default input video and output directory."""
        # Set default video if it exists
        if os.path.exists(DEFAULT_VIDEO):
            self.input_video_path = os.path.abspath(DEFAULT_VIDEO)
            self.video_path_label.setText(DEFAULT_VIDEO)
            self.load_video(self.input_video_path)
            self.update_status(f"Default video loaded: {DEFAULT_VIDEO}")
        else:
            self.update_status(
                f"Default video not found: {DEFAULT_VIDEO}. Please select a video.",
                "warning"
            )

        # Set default output directory
        self.output_dir = os.path.abspath(DEFAULT_OUTPUT_DIR)
        self.output_dir_label.setText(DEFAULT_OUTPUT_DIR)
        self.update_status(f"Default output directory: {DEFAULT_OUTPUT_DIR}")

        # Enable run button if video is loaded
        self.update_button_states()

    def browse_video(self):
        """Open file dialog to select input video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )

        if file_path:
            self.input_video_path = file_path
            self.video_path_label.setText(file_path)
            self.load_video(file_path)
            self.update_status(f"Selected video: {os.path.basename(file_path)}")
            self.update_button_states()

    def browse_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir or ""
        )

        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.update_status(f"Output directory: {dir_path}")
            self.update_button_states()

    def load_video(self, video_path: str):
        """Load video file for playback."""
        try:
            # Release previous video if any
            if self.video_capture:
                self.video_capture.release()

            # Open video
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # Get video properties
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

            # Reset playback state
            self.current_frame_idx = 0
            self.is_playing = False

            # Update slider
            self.frame_slider.setMaximum(max(0, self.total_frames - 1))
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)

            # Enable controls
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)

            # Display first frame
            self.display_frame(0)

            logger.info(
                f"Video loaded: {video_path} ({self.total_frames} frames, {self.fps:.2f} fps)"
            )

        except Exception as e:
            self.update_status(f"Failed to load video: {e}", "error")
            logger.exception(f"Video load error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load video:\n\n{e}")

    def display_frame(self, frame_idx: int):
        """Display specific frame with overlay if available."""
        if not self.video_capture:
            return

        try:
            # Seek to frame
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_capture.read()

            if not ret:
                return

            self.current_frame = frame.copy()
            self.current_frame_idx = frame_idx

            # Draw overlay if metadata available
            if self.metadata:
                frame = self.draw_overlay(frame, frame_idx)

            # Convert to QImage for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale to fit display area while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            # Update frame counter
            self.frame_counter_label.setText(f"Frame: {frame_idx} / {self.total_frames - 1}")

        except Exception as e:
            logger.exception(f"Frame display error: {e}")

    def draw_overlay(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw bounding box overlay for current frame."""
        if not self.metadata or "per_frame_boxes" not in self.metadata:
            return frame

        # Find box for current frame
        per_frame_boxes = self.metadata["per_frame_boxes"]

        for box_data in per_frame_boxes:
            if box_data["frame_idx"] == frame_idx:
                box = box_data["box"]
                x1, y1 = int(box["x1"]), int(box["y1"])
                x2, y2 = int(box["x2"]), int(box["y2"])

                # Draw green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw frame number
                label = f"Frame {frame_idx}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                break

        return frame

    def toggle_playback(self):
        """Toggle play/pause state."""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start video playback."""
        if not self.video_capture:
            return

        self.is_playing = True
        self.play_btn.setText("Pause")

        # Calculate timer interval in milliseconds
        interval = int(1000 / self.fps)
        self.playback_timer.start(interval)

        self.update_status("Playback started")

    def pause_playback(self):
        """Pause video playback."""
        self.is_playing = False
        self.play_btn.setText("Play")
        self.playback_timer.stop()
        self.update_status("Playback paused")

    def stop_playback(self):
        """Stop playback and return to first frame."""
        self.pause_playback()
        self.display_frame(0)
        self.frame_slider.setValue(0)
        self.update_status("Playback stopped")

    def update_frame(self):
        """Update to next frame during playback."""
        if not self.is_playing:
            return

        next_frame = self.current_frame_idx + 1

        if next_frame >= self.total_frames:
            # Reached end of video
            self.stop_playback()
        else:
            self.display_frame(next_frame)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(next_frame)
            self.frame_slider.blockSignals(False)

    def seek_frame(self, frame_idx: int):
        """Seek to specific frame."""
        if self.video_capture and not self.is_playing:
            self.display_frame(frame_idx)

    def update_button_states(self):
        """Update button enabled states based on current state."""
        has_video = self.input_video_path and os.path.exists(self.input_video_path)
        has_output = self.output_dir is not None
        has_results = self.pipeline_results is not None

        self.run_pipeline_btn.setEnabled(has_video and has_output)
        self.export_overlay_btn.setEnabled(has_results)

    def run_pipeline(self):
        """Run post-processing pipeline on selected video."""
        if not self.input_video_path or not self.output_dir:
            QMessageBox.warning(
                self, "Missing Input",
                "Please select both input video and output directory."
            )
            return

        if not os.path.exists(self.input_video_path):
            QMessageBox.critical(
                self, "Error",
                f"Input video does not exist:\n{self.input_video_path}"
            )
            return

        # Confirm action
        reply = QMessageBox.question(
            self,
            "Run Pipeline",
            f"Run post-processing pipeline on:\n{os.path.basename(self.input_video_path)}\n\n"
            f"Output directory:\n{self.output_dir}\n\n"
            "This may take several minutes. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        # Disable buttons during processing
        self.run_pipeline_btn.setEnabled(False)
        self.browse_video_btn.setEnabled(False)
        self.browse_output_btn.setEnabled(False)

        self.update_status("=" * 60)
        self.update_status("Starting post-processing pipeline...")

        # Create and start worker thread
        self.worker = PipelineWorker(self.input_video_path, self.output_dir)
        self.worker.finished.connect(self.on_pipeline_success)
        self.worker.error.connect(self.on_pipeline_error)
        self.worker.progress.connect(self.update_status)
        self.worker.start()

    def on_pipeline_success(self, results: Dict[str, Any]):
        """Handle successful pipeline completion."""
        self.worker = None
        self.pipeline_results = results

        # Re-enable buttons
        self.run_pipeline_btn.setEnabled(True)
        self.browse_video_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.export_overlay_btn.setEnabled(True)

        # Load metadata
        output_json = results.get("output_json")
        if output_json and os.path.exists(output_json):
            try:
                with open(output_json, 'r') as f:
                    self.metadata = json.load(f)

                # Reload current frame to show overlay
                if self.video_capture:
                    self.display_frame(self.current_frame_idx)

            except Exception as e:
                self.update_status(f"Failed to load metadata: {e}", "error")

        # Display results
        self.update_status("=" * 60)
        self.update_status("Pipeline completed successfully!")
        self.update_status(f"Total frames: {results.get('total_frames', 'N/A')}")
        self.update_status(f"Detections: {results.get('num_detections', 'N/A')}")
        self.update_status(f"Trajectory length: {results.get('trajectory_length', 'N/A')}")

        if "fixed_box_size" in results:
            box_size = results["fixed_box_size"]
            self.update_status(
                f"Fixed box size: {box_size['width']:.1f}x{box_size['height']:.1f}"
            )

        self.update_status(f"Output JSON: {results.get('output_json', 'N/A')}")

        if "output_video" in results:
            self.update_status(f"Output video: {results['output_video']}")

        self.update_status("=" * 60)

        # Show success dialog
        QMessageBox.information(
            self,
            "Success",
            f"Pipeline completed successfully!\n\n"
            f"Detections: {results.get('num_detections', 'N/A')}\n"
            f"Trajectory points: {results.get('trajectory_length', 'N/A')}\n\n"
            f"Output files saved to:\n{self.output_dir}"
        )

    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline error."""
        self.worker = None

        # Re-enable buttons
        self.run_pipeline_btn.setEnabled(True)
        self.browse_video_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)

        self.update_status("=" * 60)
        self.update_status(f"Pipeline failed: {error_msg}", "error")
        self.update_status("=" * 60)

        QMessageBox.critical(
            self,
            "Pipeline Error",
            f"Pipeline execution failed:\n\n{error_msg}\n\n"
            "Please check the log file for details."
        )

    def export_overlay_image(self):
        """Export last frame with all bounding boxes overlaid in green."""
        if not self.metadata or "per_frame_boxes" not in self.metadata:
            QMessageBox.warning(
                self,
                "No Data",
                "No pipeline results available. Please run the pipeline first."
            )
            return

        if not self.video_capture:
            QMessageBox.warning(self, "No Video", "No video loaded.")
            return

        try:
            # Get last frame
            last_frame_idx = self.total_frames - 1
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
            ret, frame = self.video_capture.read()

            if not ret:
                raise ValueError("Failed to read last frame")

            # Draw all bounding boxes in green
            per_frame_boxes = self.metadata["per_frame_boxes"]
            for box_data in per_frame_boxes:
                box = box_data["box"]
                x1, y1 = int(box["x1"]), int(box["y1"])
                x2, y2 = int(box["x2"]), int(box["y2"])

                # Draw green rectangle with some transparency effect
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"overlay_all_boxes_{timestamp}.png")

            # Save image
            cv2.imwrite(output_path, frame)

            self.update_status(f"Exported overlay image: {output_path}")
            logger.info(f"Overlay image exported: {output_path}")

            # Show success dialog
            reply = QMessageBox.information(
                self,
                "Export Successful",
                f"Overlay image exported successfully!\n\n"
                f"All {len(per_frame_boxes)} bounding boxes drawn on last frame.\n\n"
                f"Saved to:\n{output_path}\n\n"
                "Open the file?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                # Open file with default application
                import subprocess
                if sys.platform == "win32":
                    os.startfile(output_path)
                elif sys.platform == "darwin":
                    subprocess.call(["open", output_path])
                else:
                    subprocess.call(["xdg-open", output_path])

        except Exception as e:
            self.update_status(f"Failed to export overlay image: {e}", "error")
            logger.exception(f"Overlay export error: {e}")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export overlay image:\n\n{e}"
            )

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Application closing")

        # Stop playback
        if self.is_playing:
            self.pause_playback()

        # Release video capture
        if self.video_capture:
            self.video_capture.release()

        # Wait for worker thread
        if self.worker and self.worker.isRunning():
            logger.info("Waiting for pipeline to complete...")
            self.worker.wait(5000)

        event.accept()


def main():
    """Main entry point for the application."""
    logger.info("=" * 80)
    logger.info("Post-Processing GUI starting")
    logger.info("=" * 80)

    app = QApplication(sys.argv)
    app.setApplicationName("Post-Processing Pipeline Testing Tool")
    app.setStyle("Fusion")

    try:
        window = PostProcessingGUI()
        window.show()

        exit_code = app.exec()
        logger.info(f"Application exited with code {exit_code}")
        return exit_code

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"A fatal error occurred:\n\n{e}\n\n"
            "Please check post_processing_gui.log for details."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
