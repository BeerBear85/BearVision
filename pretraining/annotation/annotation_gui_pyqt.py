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
    QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont

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
    ap.run(cfg, show_preview=False, frame_callback=frame_callback)


class AnnotationGUI(QMainWindow):
    """PySide6 front-end for the annotation pipeline."""

    def __init__(self):
        """Create widgets and bind actions."""
        super().__init__()
        self.setWindowTitle("Annotation Pipeline")
        self.setGeometry(100, 100, 900, 400)
        
        # Load configuration so preview scaling can be adjusted via YAML without
        # touching the code. Path resolution stays relative to this file.
        cfg_path = Path(__file__).with_name("sample_config.yaml")
        self.base_cfg = ap._ensure_cfg(str(cfg_path))
        self.preview_scaling = self.base_cfg.preview_scaling
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
        
        # Add left panel to splitter
        main_splitter.addWidget(left_panel)
        
        # Right panel - Image preview (150px wide, flexible height)
        self.preview_panel = QFrame()
        self.preview_panel.setFixedWidth(150)
        self.preview_panel.setStyleSheet("background-color: lightgray; border: 1px solid gray;")
        preview_layout = QVBoxLayout(self.preview_panel)
        
        preview_label = QLabel("Image Preview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_layout.addWidget(preview_label)
        
        placeholder_label = QLabel("(Processed frames will appear in separate OpenCV window)")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setWordWrap(True)
        placeholder_label.setStyleSheet("color: gray; font-style: italic;")
        preview_layout.addWidget(placeholder_label)
        
        # Add preview panel to splitter
        main_splitter.addWidget(self.preview_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([750, 150])
        
        # Initialize variables
        self.video_path = ""
        self.output_dir = ""
        self._preview_window = None
        
        # Set up frame signals for thread-safe updates
        self.frame_signals = FrameSignals()
        self.frame_signals.frame_ready.connect(self._update_preview)
        
        # Timer for status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.refresh_status)
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
            # Re-enable the run button and close preview window
            self.run_btn.setEnabled(True)
            self._close_preview_window()

    def on_frame(self, frame: np.ndarray) -> None:
        """Schedule display of a processed frame on the main thread."""
        self.frame_signals.frame_ready.emit(frame.copy())

    def _update_preview(self, frame: np.ndarray) -> None:
        """Render a scaled preview image in a separate OpenCV window."""
        # Lazily create the window to avoid opening it during app start when no
        # frames have been processed yet.
        if self._preview_window is None:
            self._preview_window = "preview"
            # ``WINDOW_NORMAL`` allows user-resizable window for convenience.
            cv2.namedWindow(self._preview_window, cv2.WINDOW_NORMAL)

        # Downscale the frame so high-resolution videos don't overwhelm the
        # display and to keep per-frame processing lightweight.
        h, w = frame.shape[:2]
        scaled = cv2.resize(
            frame, 
            (int(w * self.preview_scaling), int(h * self.preview_scaling))
        )
        cv2.imshow(self._preview_window, scaled)
        # ``waitKey`` with small timeout lets OpenCV process its event queue
        # without noticeably blocking the Qt loop.
        cv2.waitKey(1)

    def _close_preview_window(self) -> None:
        """Destroy the OpenCV preview window if it was created."""
        if self._preview_window is not None:
            cv2.destroyWindow(self._preview_window)
            self._preview_window = None

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


def create_app():
    """Create and return a QApplication instance for headless testing."""
    return QApplication(sys.argv if sys.argv else [])


def main():
    """Entry point for running the annotation GUI."""
    app = create_app()
    
    window = AnnotationGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()