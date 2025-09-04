"""
GoPro Manual Test GUI

A simple GUI application for manually testing GoPro Black 12 preview functionality.
Provides a Start Preview button to display GoPro camera preview streams.
"""

import sys
import os
import threading
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QPixmap, QImage, QFont

# Add module path for GoProController
MODULE_DIR = Path(__file__).resolve().parent.parent / "code" / "modules"
sys.path.append(str(MODULE_DIR))

from GoProController import GoProController


class PreviewWorker(QThread):
    """Worker thread for handling GoPro preview stream."""
    
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self, stream_url):
        super().__init__()
        self.stream_url = stream_url
        self.cap = None
        self.running = False
        
    def run(self):
        """Start capturing frames from the GoPro preview stream."""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Failed to open stream: {self.stream_url}")
                return
                
            self.running = True
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    self.error_occurred.emit("Failed to read frame from stream")
                    break
                    
        except Exception as e:
            self.error_occurred.emit(f"Stream error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
                
    def stop(self):
        """Stop the preview stream."""
        self.running = False
        self.wait()


class GoProManualTestGUI(QMainWindow):
    """Main GUI window for GoPro manual testing."""
    
    def __init__(self):
        super().__init__()
        self.gopro_controller = None
        self.preview_worker = None
        self.preview_active = False
        
        self.setWindowTitle("GoPro Manual Test GUI")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("GoPro Black 12 Manual Test")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Control panel
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        
        self.start_preview_btn = QPushButton("Start Preview")
        self.start_preview_btn.setMinimumHeight(40)
        self.start_preview_btn.clicked.connect(self.toggle_preview)
        control_layout.addWidget(self.start_preview_btn)
        
        control_layout.addStretch()
        main_layout.addWidget(control_frame)
        
        # Preview display area
        self.preview_label = QLabel("GoPro preview will appear here")
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                background-color: #f0f0f0;
                color: #666666;
                font-size: 14px;
            }
        """)
        main_layout.addWidget(self.preview_label)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666666; font-size: 12px; padding: 5px;")
        main_layout.addWidget(self.status_label)
        
    def toggle_preview(self):
        """Toggle the GoPro preview stream on/off."""
        if not self.preview_active:
            self.start_gopro_preview()
        else:
            self.stop_gopro_preview()
            
    def start_gopro_preview(self):
        """Start the GoPro preview stream."""
        try:
            self.status_label.setText("Connecting to GoPro...")
            self.start_preview_btn.setEnabled(False)
            
            # Initialize and connect to GoPro
            self.gopro_controller = GoProController()
            self.gopro_controller.connect()
            
            # Start preview stream
            self.status_label.setText("Starting preview stream...")
            stream_url = self.gopro_controller.start_preview()
            
            # Start the preview worker thread
            self.preview_worker = PreviewWorker(stream_url)
            self.preview_worker.frame_ready.connect(self.update_preview_frame)
            self.preview_worker.error_occurred.connect(self.handle_preview_error)
            self.preview_worker.start()
            
            # Update UI state
            self.preview_active = True
            self.start_preview_btn.setText("Stop Preview")
            self.start_preview_btn.setEnabled(True)
            self.status_label.setText("Preview active")
            
        except Exception as e:
            self.handle_preview_error(f"Failed to start preview: {str(e)}")
            
    def stop_gopro_preview(self):
        """Stop the GoPro preview stream."""
        try:
            self.status_label.setText("Stopping preview...")
            
            # Stop preview worker
            if self.preview_worker:
                self.preview_worker.stop()
                self.preview_worker = None
                
            # Disconnect from GoPro
            if self.gopro_controller:
                self.gopro_controller.disconnect()
                self.gopro_controller = None
                
            # Reset UI
            self.preview_active = False
            self.start_preview_btn.setText("Start Preview")
            self.preview_label.setText("GoPro preview will appear here")
            self.preview_label.setPixmap(QPixmap())  # Clear any existing image
            self.status_label.setText("Ready")
            
        except Exception as e:
            self.show_error_popup(f"Error stopping preview: {str(e)}")
            
    def update_preview_frame(self, frame):
        """Update the preview display with a new frame."""
        try:
            # Convert OpenCV frame (BGR) to Qt format (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            bytes_per_line = channels * width
            
            # Create QImage and QPixmap
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit preview label while maintaining aspect ratio
            label_size = self.preview_label.size()
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.handle_preview_error(f"Failed to update preview frame: {str(e)}")
            
    def handle_preview_error(self, error_message):
        """Handle preview stream errors."""
        self.show_error_popup(error_message)
        self.stop_gopro_preview()
        
    def show_error_popup(self, message):
        """Show an error popup dialog."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("GoPro Preview Error")
        msg_box.setText("Failed to start/maintain GoPro preview.")
        msg_box.setDetailedText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()
        
    def closeEvent(self, event):
        """Clean up when the window is closed."""
        if self.preview_active:
            self.stop_gopro_preview()
        event.accept()


def main():
    """Main entry point for the GoPro Manual Test GUI."""
    app = QApplication(sys.argv)
    window = GoProManualTestGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()