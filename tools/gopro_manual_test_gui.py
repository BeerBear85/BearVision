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


class PingTestWorker(QThread):
    """Worker thread for testing GoPro connectivity."""
    
    ping_success = Signal()
    ping_failed = Signal(str)
    
    def __init__(self):
        super().__init__()
        
    def run(self):
        """Test GoPro connectivity in background thread."""
        try:
            test_controller = GoProController()
            test_controller.connect()
            test_controller.disconnect()
            self.ping_success.emit()
        except Exception as e:
            self.ping_failed.emit(str(e))


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
        self.ping_worker = None
        self.preview_active = False
        self.recording_active = False
        
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
        control_layout = QVBoxLayout(control_frame)
        
        # First row of buttons
        first_row_layout = QHBoxLayout()
        
        self.ping_test_btn = QPushButton("Ping Test")
        self.ping_test_btn.setMinimumHeight(40)
        self.ping_test_btn.clicked.connect(self.run_ping_test)
        first_row_layout.addWidget(self.ping_test_btn)
        
        self.start_preview_btn = QPushButton("Start Preview")
        self.start_preview_btn.setMinimumHeight(40)
        self.start_preview_btn.clicked.connect(self.toggle_preview)
        first_row_layout.addWidget(self.start_preview_btn)
        
        first_row_layout.addStretch()
        control_layout.addLayout(first_row_layout)
        
        # Second row of buttons
        second_row_layout = QHBoxLayout()
        
        self.hindsight_btn = QPushButton("Capture Hindsight (15s)")
        self.hindsight_btn.setMinimumHeight(40)
        self.hindsight_btn.clicked.connect(self.capture_hindsight)
        self.hindsight_btn.setEnabled(False)  # Enable only when connected
        second_row_layout.addWidget(self.hindsight_btn)
        
        self.recording_btn = QPushButton("Start Recording")
        self.recording_btn.setMinimumHeight(40)
        self.recording_btn.clicked.connect(self.toggle_recording)
        self.recording_btn.setEnabled(False)  # Enable only when connected
        second_row_layout.addWidget(self.recording_btn)
        
        second_row_layout.addStretch()
        control_layout.addLayout(second_row_layout)
        
        # Recording status indicator
        self.recording_status_label = QLabel("● Not Recording")
        self.recording_status_label.setAlignment(Qt.AlignCenter)
        self.recording_status_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
        """)
        control_layout.addWidget(self.recording_status_label)
        
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
        
    def run_ping_test(self):
        """Test basic GoPro connectivity in a background thread."""
        if self.ping_worker and self.ping_worker.isRunning():
            return  # Already running
            
        self.status_label.setText("Testing GoPro connection...")
        self.ping_test_btn.setEnabled(False)
        self.start_preview_btn.setEnabled(False)
        
        # Create and start ping test worker
        self.ping_worker = PingTestWorker()
        self.ping_worker.ping_success.connect(self.on_ping_success)
        self.ping_worker.ping_failed.connect(self.on_ping_failed)
        self.ping_worker.start()
        
    def on_ping_success(self):
        """Handle successful ping test."""
        self.status_label.setText("Ping test successful - GoPro is reachable")
        self.show_info_popup("Connection Test", "✅ GoPro connection successful!\n\nThe camera is reachable and responding to commands.")
        self.ping_test_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(True)
        self.hindsight_btn.setEnabled(True)
        self.recording_btn.setEnabled(True)
        
    def on_ping_failed(self, error_message):
        """Handle failed ping test."""
        self.status_label.setText("Ping test failed")
        self.show_error_popup(f"Ping test failed: {error_message}")
        self.ping_test_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(True)
        self.hindsight_btn.setEnabled(False)
        self.recording_btn.setEnabled(False)
        
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
                
            # Stop preview stream and disconnect from GoPro
            if self.gopro_controller:
                try:
                    self.gopro_controller.stop_preview()
                except:
                    pass  # Ignore errors when stopping preview
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
        
    def show_info_popup(self, title, message):
        """Show an info popup dialog."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def capture_hindsight(self):
        """Capture a 15-second HindSight clip."""
        try:
            if not self.gopro_controller:
                self.gopro_controller = GoProController()
                self.gopro_controller.connect()
                
            self.status_label.setText("Capturing HindSight clip...")
            self.hindsight_btn.setEnabled(False)
            
            # Run hindsight capture in background thread
            def run_hindsight():
                try:
                    self.gopro_controller.startHindsightMode()  # Use simplified hindsight mode
                    QTimer.singleShot(0, lambda: self.on_hindsight_success())
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_hindsight_failed(str(e)))
            
            threading.Thread(target=run_hindsight, daemon=True).start()
            
        except Exception as e:
            self.on_hindsight_failed(str(e))
    
    def on_hindsight_success(self):
        """Handle successful hindsight capture."""
        self.status_label.setText("HindSight clip captured successfully")
        self.show_info_popup("HindSight Capture", "✅ 15-second HindSight clip captured successfully!")
        self.hindsight_btn.setEnabled(True)
    
    def on_hindsight_failed(self, error_message):
        """Handle failed hindsight capture."""
        self.status_label.setText("HindSight capture failed")
        self.show_error_popup(f"HindSight capture failed: {error_message}")
        self.hindsight_btn.setEnabled(True)

    def toggle_recording(self):
        """Toggle video recording on/off."""
        if not self.recording_active:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start video recording."""
        try:
            if not self.gopro_controller:
                self.gopro_controller = GoProController()
                self.gopro_controller.connect()
                
            self.status_label.setText("Starting recording...")
            self.recording_btn.setEnabled(False)
            
            # Run recording start in background thread
            def run_start_recording():
                try:
                    self.gopro_controller.start_recording()
                    QTimer.singleShot(0, lambda: self.on_recording_started())
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_recording_failed(str(e)))
            
            threading.Thread(target=run_start_recording, daemon=True).start()
            
        except Exception as e:
            self.on_recording_failed(str(e))
    
    def stop_recording(self):
        """Stop video recording."""
        try:
            if not self.gopro_controller:
                return
                
            self.status_label.setText("Stopping recording...")
            self.recording_btn.setEnabled(False)
            
            # Run recording stop in background thread
            def run_stop_recording():
                try:
                    self.gopro_controller.stop_recording()
                    QTimer.singleShot(0, lambda: self.on_recording_stopped())
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_recording_failed(str(e)))
            
            threading.Thread(target=run_stop_recording, daemon=True).start()
            
        except Exception as e:
            self.on_recording_failed(str(e))
    
    def on_recording_started(self):
        """Handle successful recording start."""
        self.recording_active = True
        self.recording_btn.setText("Stop Recording")
        self.recording_btn.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold;")
        self.recording_btn.setEnabled(True)
        self.recording_status_label.setText("● Recording")
        self.recording_status_label.setStyleSheet("""
            QLabel {
                color: #ff0000;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                border: 1px solid #ff4444;
                border-radius: 5px;
                background-color: #ffe6e6;
            }
        """)
        self.status_label.setText("Recording started")
    
    def on_recording_stopped(self):
        """Handle successful recording stop."""
        self.recording_active = False
        self.recording_btn.setText("Start Recording")
        self.recording_btn.setStyleSheet("")  # Reset to default style
        self.recording_btn.setEnabled(True)
        self.recording_status_label.setText("● Not Recording")
        self.recording_status_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
        """)
        self.status_label.setText("Recording stopped")
    
    def on_recording_failed(self, error_message):
        """Handle failed recording operation."""
        self.show_error_popup(f"Recording operation failed: {error_message}")
        self.recording_btn.setEnabled(True)
        self.status_label.setText("Recording operation failed")
        
    def closeEvent(self, event):
        """Clean up when the window is closed."""
        if self.preview_active:
            self.stop_gopro_preview()
        if self.ping_worker and self.ping_worker.isRunning():
            self.ping_worker.quit()
            self.ping_worker.wait()
        event.accept()


def main():
    """Main entry point for the GoPro Manual Test GUI."""
    app = QApplication(sys.argv)
    window = GoProManualTestGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()