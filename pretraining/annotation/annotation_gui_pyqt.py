"""PySide6-based front-end for the annotation pipeline."""

import threading
import sys
import os
import copy
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QFileDialog, QMessageBox,
    QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QPixmap, QImage

import annotation_pipeline as ap


class FrameSignals(QObject):
    """Signals for thread-safe frame updates."""
    frame_ready = Signal(np.ndarray)


def run_pipeline(
    cfg: ap.PipelineConfig,
    video_path: str,
    output_dir: str,
    frame_callback: Callable[[np.ndarray], None],
) -> None:
    """Execute the pipeline for one video and export a dataset.

    Purpose
    -------
    Reuse a base configuration and invoke :func:`annotation_pipeline.run` so
    that frames, labels and interpolated trajectories are generated for the
    chosen video while emitting per-frame previews.

    Inputs
    ------
    cfg: ap.PipelineConfig
        Base configuration object which will be copied to avoid mutating
        caller state.
    video_path: str
        Path to the source video file.
    output_dir: str
        Directory where the dataset will be written.
    frame_callback: Callable[[np.ndarray], None]
        Function receiving each processed frame for preview rendering.

    Outputs
    -------
    None
        Files are written to ``output_dir`` and ``frame_callback`` is invoked
        for every exported frame.
    """
    # Reset progress so a new run starts with a clean status snapshot.
    ap.status = ap.PipelineStatus()
    cfg = copy.deepcopy(cfg)
    cfg.videos = [video_path]
    cfg.export.output_dir = output_dir
    # The callback feeds the GUI a copy of each frame.  We set
    # ``show_preview`` to ``False`` because the GUI now controls preview
    # rendering in its own OpenCV window rather than relying on the pipeline's
    # post-run display.
    ap.run(cfg, show_preview=False, frame_callback=frame_callback, gui_mode=True)


class AnnotationGUI(QMainWindow):
    """PySide6 front-end for the annotation pipeline."""

    def __init__(self):
        """Create widgets and bind actions."""
        super().__init__()
        self.setWindowTitle("Annotation Pipeline")
        self.setGeometry(100, 100, 900, 400)
        
        # Load configuration for pipeline parameters
        cfg_path = Path(__file__).with_name("sample_config.yaml")
        self.base_cfg = ap._ensure_cfg(str(cfg_path))
        # Get GUI configuration values
        gui_cfg = self.base_cfg.gui
        self.preview_width = gui_cfg.preview_width
        self.preview_panel_width = gui_cfg.preview_panel_width
        self.preview_image_min_height = gui_cfg.preview_image_min_height
        self.trajectory_image_min_height = gui_cfg.trajectory_image_min_height
        ap.status = ap.PipelineStatus()  # Reset status so GUI starts in "Idle".
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create main splitter for layout with preview panel
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video selection
        video_layout = QHBoxLayout()
        self.select_video_btn = QPushButton("Select Video")
        self.select_video_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.select_video_btn)
        left_layout.addLayout(video_layout)
        
        self.video_path_label = QLabel("No video selected")
        self.video_path_label.setWordWrap(True)
        left_layout.addWidget(self.video_path_label)
        
        # Output selection
        output_layout = QHBoxLayout()
        self.select_output_btn = QPushButton("Select Output")
        self.select_output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.select_output_btn)
        left_layout.addLayout(output_layout)
        
        self.output_dir_label = QLabel("No output directory selected")
        self.output_dir_label.setWordWrap(True)
        left_layout.addWidget(self.output_dir_label)
        
        # Run button
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.start)
        left_layout.addWidget(self.run_btn)
        
        # Status labels
        self.status_label = QLabel("Idle")
        left_layout.addWidget(self.status_label)
        
        self.frame_progress_label = QLabel("Processed: 0/0 frames")
        left_layout.addWidget(self.frame_progress_label)
        
        self.riders_count_label = QLabel("Riders detected: 0")
        left_layout.addWidget(self.riders_count_label)
        
        # Add left panel to splitter
        main_splitter.addWidget(left_panel)
        
        # Right panel - Image previews (configurable width, flexible height)
        self.preview_panel = QFrame()
        self.preview_panel.setFixedWidth(self.preview_panel_width)
        self.preview_panel.setStyleSheet("background-color: lightgray; border: 1px solid gray;")
        preview_layout = QVBoxLayout(self.preview_panel)
        
        # Frame-by-frame preview section
        frame_preview_label = QLabel("Frame Preview")
        frame_preview_label.setAlignment(Qt.AlignCenter)
        frame_preview_label.setWordWrap(True)
        preview_layout.addWidget(frame_preview_label)
        
        # Create scrollable area for the frame preview
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        
        # Image label that will display the frames
        self.preview_image_label = QLabel("No frames processed yet")
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setStyleSheet("color: gray; font-style: italic;")
        self.preview_image_label.setMinimumSize(self.preview_width, self.preview_image_min_height)
        
        self.preview_scroll.setWidget(self.preview_image_label)
        preview_layout.addWidget(self.preview_scroll)
        
        # Trajectory preview section
        trajectory_preview_label = QLabel("Trajectory Preview")
        trajectory_preview_label.setAlignment(Qt.AlignCenter)
        trajectory_preview_label.setWordWrap(True)
        preview_layout.addWidget(trajectory_preview_label)
        
        # Create scrollable area for the trajectory preview
        self.trajectory_scroll = QScrollArea()
        self.trajectory_scroll.setWidgetResizable(True)
        self.trajectory_scroll.setAlignment(Qt.AlignCenter)
        
        # Image label that will display the trajectory
        self.trajectory_image_label = QLabel("No trajectory generated yet")
        self.trajectory_image_label.setAlignment(Qt.AlignCenter)
        self.trajectory_image_label.setStyleSheet("color: gray; font-style: italic;")
        self.trajectory_image_label.setMinimumSize(self.preview_width, self.trajectory_image_min_height)
        
        self.trajectory_scroll.setWidget(self.trajectory_image_label)
        preview_layout.addWidget(self.trajectory_scroll)
        
        # Add preview panel to splitter
        main_splitter.addWidget(self.preview_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, self.preview_panel_width])
        
        # Initialize variables with config defaults (convert to absolute paths)
        project_root = Path(__file__).resolve().parents[2]  # Go up to BearVision_2 root
        if gui_cfg.default_video_path:
            self.video_path = str(project_root / gui_cfg.default_video_path)
        else:
            self.video_path = ""
        if gui_cfg.default_output_dir:
            self.output_dir = str(project_root / gui_cfg.default_output_dir)
        else:
            self.output_dir = ""
        self.last_trajectory_path = None
        
        # Update labels with default values if they exist
        if self.video_path:
            self.video_path_label.setText(self.video_path)
        if self.output_dir:
            self.output_dir_label.setText(self.output_dir)
        
        # Set up frame signals for thread-safe updates
        self.frame_signals = FrameSignals()
        self.frame_signals.frame_ready.connect(self._update_preview)
        
        # Timer for status updates and trajectory monitoring
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.refresh_status)
        self.status_timer.timeout.connect(self._check_trajectory_updates)
        self.status_timer.start(200)  # Update every 200ms

    def select_video(self) -> None:
        """Prompt the user to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
        )
        if file_path:
            self.video_path = file_path
            self.video_path_label.setText(file_path)

    def select_output(self) -> None:
        """Prompt the user to select an output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output directory"
        )
        if directory:
            self.output_dir = directory
            self.output_dir_label.setText(directory)

    def start(self) -> None:
        """Launch the pipeline in a background thread."""
        if not self.video_path or not self.output_dir:
            QMessageBox.critical(
                self,
                "Missing paths",
                "Please select video and output directory"
            )
            return
            
        # Disable the button immediately so accidental double-clicks do not
        # spawn multiple pipeline instances competing for resources.
        self.run_btn.setEnabled(False)
        
        # Running in a thread keeps the GUI responsive during processing.
        thread = threading.Thread(
            target=self._run_pipeline_thread, 
            args=(self.video_path, self.output_dir), 
            daemon=True
        )
        thread.start()

    def _run_pipeline_thread(self, video: str, output: str) -> None:
        """Execute the pipeline and re-enable the Run button when done."""
        try:
            run_pipeline(self.base_cfg, video, output, self.on_frame)
        except Exception as exc:
            # Without this handler users receive no feedback when the pipeline
            # fails (e.g. due to missing video/weights).  Showing a message box
            # makes the failure explicit and keeps the GUI responsive.
            QMessageBox.critical(self, "Pipeline error", str(exc))
        finally:
            # Re-enable the run button
            self.run_btn.setEnabled(True)

    def on_frame(self, frame: np.ndarray) -> None:
        """Schedule display of a processed frame on the main thread."""
        self.frame_signals.frame_ready.emit(frame.copy())

    def _update_preview(self, frame: np.ndarray) -> None:
        """Render a scaled preview image in the Qt widget."""
        # Calculate preview scaling based on the preview panel width
        # to ensure the preview fits nicely in the GUI
        h, w = frame.shape[:2]
        scale_factor = self.preview_width / w if w > self.preview_width else 1.0
        scaled = cv2.resize(
            frame, 
            (int(w * scale_factor), int(h * scale_factor))
        )
        
        # Convert BGR to RGB for Qt display
        rgb_frame = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qt_image)
        self.preview_image_label.setPixmap(pixmap)
        self.preview_image_label.setScaledContents(False)
        
        # Adjust the label size to fit the image
        self.preview_image_label.resize(pixmap.size())

    def _check_trajectory_updates(self) -> None:
        """Check for new trajectory images and update the trajectory preview."""
        if not self.output_dir:
            return
            
        # Look for trajectory images in the output directory
        trajectory_dir = Path(self.output_dir) / "trajectories"
        if not trajectory_dir.exists():
            return
            
        # Find the most recent trajectory image
        trajectory_files = list(trajectory_dir.glob("trajectory_*.jpg"))
        if not trajectory_files:
            return
            
        # Sort by modification time, get the newest
        latest_trajectory = max(trajectory_files, key=lambda p: p.stat().st_mtime)
        
        # Only update if this is a new trajectory image
        if self.last_trajectory_path != str(latest_trajectory):
            self.last_trajectory_path = str(latest_trajectory)
            self._update_trajectory_preview(str(latest_trajectory))

    def _update_trajectory_preview(self, trajectory_path: str) -> None:
        """Update the trajectory preview with the latest trajectory image."""
        try:
            # Load the trajectory image
            trajectory_img = cv2.imread(trajectory_path)
            if trajectory_img is None:
                return
                
            # Scale the image to fit the preview panel
            h, w = trajectory_img.shape[:2]
            scale_factor = self.preview_width / w if w > self.preview_width else 1.0
            scaled = cv2.resize(
                trajectory_img, 
                (int(w * scale_factor), int(h * scale_factor))
            )
            
            # Convert BGR to RGB for Qt display
            rgb_frame = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(qt_image)
            self.trajectory_image_label.setPixmap(pixmap)
            self.trajectory_image_label.setScaledContents(False)
            
            # Adjust the label size to fit the image
            self.trajectory_image_label.resize(pixmap.size())
            
        except Exception as e:
            # If there's an error loading the image, show an error message
            self.trajectory_image_label.setText(f"Error loading trajectory: {str(e)}")

    def refresh_status(self) -> None:
        """Update status and frame-progress labels with pipeline progress."""
        st = ap.status
        # Always show the last function name; fall back to "Idle" when nothing
        # has run yet for clarity at application start.
        self.status_label.setText(st.last_function or "Idle")
        
        if st.total_frames:
            # Present progress as "processed/total" to clarify that sampling may
            # result in the first value being lower than the total frame count.
            self.frame_progress_label.setText(
                f"Processed: {st.current_frame}/{st.total_frames} frames"
            )
        else:
            self.frame_progress_label.setText("Processed: 0/0 frames")
            
        # Update rider counter display
        self.riders_count_label.setText(f"Riders detected: {st.riders_detected}")


def create_app():
    """Create and return a QApplication instance for headless testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if sys.argv else [])
    return app


def main():
    """Entry point for running the annotation GUI."""
    app = create_app()
    
    window = AnnotationGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()