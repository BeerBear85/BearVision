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
    """Signals used to synchronize image updates across threads.

    Purpose
    -------
    Provide Qt ``Signal`` objects for both the per-frame preview and the
    trajectory overlay so worker threads can safely communicate with the GUI.

    Inputs
    ------
    None

    Outputs
    -------
    None
    """

    # Emitted whenever the pipeline processes a new frame. The GUI connects
    # this to a slot that renders the live preview.
    frame_ready = Signal(np.ndarray)
    # Emitted after the pipeline generates a trajectory visualization. This
    # updates a second preview so users can inspect the full path.
    trajectory_ready = Signal(np.ndarray)


def run_pipeline(
    cfg: ap.PipelineConfig,
    video_path: str,
    output_dir: str,
    frame_callback: Callable[[np.ndarray], None],
    traj_callback: Callable[[np.ndarray], None] | None = None,
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
    traj_callback: Callable[[np.ndarray], None] | None, optional
        Invoked once a trajectory image becomes available so the GUI can
        display it. ``None`` disables trajectory preview updates.

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
    ap.run(
        cfg,
        show_preview=False,
        frame_callback=frame_callback,
        gui_mode=True,
        trajectory_callback=traj_callback,
    )


class AnnotationGUI(QMainWindow):
    """PySide6 front-end for the annotation pipeline.

    Purpose
    -------
    Offer a minimal user interface for running the annotation pipeline while
    presenting live frame previews and generated trajectory overlays.

    Inputs
    ------
    None
        Configuration is supplied through GUI interactions.

    Outputs
    -------
    None
        Results are displayed within the application window and written to
        disk by the pipeline.
    """

    def __init__(self):
        """Create widgets and bind actions."""
        super().__init__()
        self.setWindowTitle("Annotation Pipeline")
        self.setGeometry(100, 100, 900, 400)
        
        # Load configuration for pipeline parameters
        cfg_path = Path(__file__).with_name("sample_config.yaml")
        self.base_cfg = ap._ensure_cfg(str(cfg_path))
        # Calculate preview scaling based on preview panel size
        self.preview_width = 280  # Maximum width for preview display
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
        
        # Right panel - Image preview (300px wide, flexible height)
        self.preview_panel = QFrame()
        self.preview_panel.setFixedWidth(300)
        self.preview_panel.setStyleSheet("background-color: lightgray; border: 1px solid gray;")
        preview_layout = QVBoxLayout(self.preview_panel)
        
        preview_label = QLabel("Image Preview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_layout.addWidget(preview_label)

        # Create scrollable area for the live frame preview. A scroll area is
        # used rather than a fixed-size label so large frames can be inspected
        # without distorting their aspect ratio.
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setAlignment(Qt.AlignCenter)

        # Image label that will display the frames processed in real time.
        self.preview_image_label = QLabel("No frames processed yet")
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setStyleSheet("color: gray; font-style: italic;")
        self.preview_image_label.setMinimumSize(280, 200)

        self.preview_scroll.setWidget(self.preview_image_label)
        preview_layout.addWidget(self.preview_scroll)

        # Trajectory preview section mirrors the live preview but updates only
        # when the backend emits a new trajectory image after track completion.
        traj_label = QLabel("Trajectory Preview")
        traj_label.setAlignment(Qt.AlignCenter)
        traj_label.setWordWrap(True)
        preview_layout.addWidget(traj_label)

        self.traj_scroll = QScrollArea()
        self.traj_scroll.setWidgetResizable(True)
        self.traj_scroll.setAlignment(Qt.AlignCenter)

        self.traj_image_label = QLabel("No trajectory available")
        self.traj_image_label.setAlignment(Qt.AlignCenter)
        self.traj_image_label.setStyleSheet("color: gray; font-style: italic;")
        self.traj_image_label.setMinimumSize(280, 200)

        self.traj_scroll.setWidget(self.traj_image_label)
        preview_layout.addWidget(self.traj_scroll)
        
        # Add preview panel to splitter
        main_splitter.addWidget(self.preview_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 300])
        
        # Initialize variables
        self.video_path = ""
        self.output_dir = ""
        
        # Set up signals for thread-safe updates. Two channels allow the GUI to
        # refresh live frames and trajectory overlays independently without
        # blocking the worker thread.
        self.frame_signals = FrameSignals()
        self.frame_signals.frame_ready.connect(self._update_preview)
        self.frame_signals.trajectory_ready.connect(self._update_trajectory_preview)
        
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
            run_pipeline(
                self.base_cfg,
                video,
                output,
                self.on_frame,
                self.on_trajectory,
            )
        except Exception as exc:
            # Without this handler users receive no feedback when the pipeline
            # fails (e.g. due to missing video/weights).  Showing a message box
            # makes the failure explicit and keeps the GUI responsive.
            QMessageBox.critical(self, "Pipeline error", str(exc))
        finally:
            # Re-enable the run button
            self.run_btn.setEnabled(True)

    def on_frame(self, frame: np.ndarray) -> None:
        """Schedule display of a processed frame on the main thread.

        Purpose
        -------
        Forward frames from the background thread to the GUI for real-time
        preview rendering.

        Inputs
        ------
        frame: np.ndarray
            BGR image of the processed frame.

        Outputs
        -------
        None
            The frame is emitted via Qt signals for display.
        """
        self.frame_signals.frame_ready.emit(frame.copy())

    def on_trajectory(self, image: np.ndarray) -> None:
        """Schedule display of the latest trajectory image.

        Purpose
        -------
        Receive a completed trajectory overlay from the worker thread and
        forward it to the GUI for rendering.

        Inputs
        ------
        image: np.ndarray
            BGR image containing the trajectory drawing.

        Outputs
        -------
        None
            The image is emitted via Qt signals for display.
        """
        # Trajectory images arrive only when a track ends, so we use a
        # dedicated signal to update the GUI without queuing behind high-rate
        # frame previews.
        self.frame_signals.trajectory_ready.emit(image.copy())

    def _update_preview(self, frame: np.ndarray) -> None:
        """Render a scaled preview image in the Qt widget.

        Purpose
        -------
        Display live pipeline frames within the GUI while preserving aspect
        ratio and fitting the predefined preview width.

        Inputs
        ------
        frame: np.ndarray
            BGR frame to display.

        Outputs
        -------
        None
            The scaled image is set on ``preview_image_label``.
        """
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

    def _update_trajectory_preview(self, image: np.ndarray) -> None:
        """Display the most recent trajectory image in the GUI.

        Purpose
        -------
        Present the complete motion path produced after a track concludes,
        allowing users to verify the interpolation quality.

        Inputs
        ------
        image: np.ndarray
            BGR image with trajectory overlay.

        Outputs
        -------
        None
            The scaled image is shown on ``traj_image_label``.
        """
        # Reuse the same scaling logic as the live preview so both panels share
        # consistent sizing and users can compare them easily.
        h, w = image.shape[:2]
        scale_factor = self.preview_width / w if w > self.preview_width else 1.0
        scaled = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))

        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        self.traj_image_label.setPixmap(pixmap)
        self.traj_image_label.setScaledContents(False)
        self.traj_image_label.resize(pixmap.size())


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