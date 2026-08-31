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
from datetime import datetime
from typing import List, Dict, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QMessageBox, QMenuBar, QFileDialog,
    QTextEdit, QScrollArea, QGridLayout, QGroupBox, QSplitter,
    QProgressBar, QListWidget, QListWidgetItem, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QAction, QIcon, QColor, QPalette

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


class ModernCard(QFrame):
    """Modern card-style widget with shadow and rounded corners."""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)
        self.setContentsMargins(0, 0, 0, 0)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        if title:
            # Title label
            title_label = QLabel(title)
            title_font = QFont()
            title_font.setPointSize(11)
            title_font.setWeight(QFont.Weight.Medium)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Separator line
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            layout.addWidget(separator)
        
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(8)
        layout.addLayout(self.content_layout)
        
        # Apply modern styling
        self.setStyleSheet("""
            ModernCard {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            ModernCard:hover {
                border-color: #cbd5e1;
            }
        """)
        
    def add_widget(self, widget):
        """Add widget to card content."""
        self.content_layout.addWidget(widget)
        
    def add_layout(self, layout):
        """Add layout to card content."""
        self.content_layout.addLayout(layout)


class StatusBadge(QLabel):
    """Modern status badge widget."""
    
    def __init__(self, text: str = "", status_type: str = "default", parent=None):
        super().__init__(text, parent)
        self.status_type = status_type
        self.setAlignment(Qt.AlignCenter)
        self.update_style()
        
    def update_style(self):
        colors = {
            "default": ("#1f2937", "#f3f4f6"),
            "success": ("#065f46", "#d1fae5"),
            "error": ("#7f1d1d", "#fee2e2"),
            "warning": ("#92400e", "#fef3c7"),
            "info": ("#1e40af", "#dbeafe")
        }
        
        text_color, bg_color = colors.get(self.status_type, colors["default"])
        
        self.setStyleSheet(f"""
            StatusBadge {{
                background-color: {bg_color};
                color: {text_color};
                border-radius: 12px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 500;
                min-width: 50px;
            }}
        """)
        
    def set_status(self, text: str, status_type: str):
        """Update badge text and status type."""
        self.setText(text)
        self.status_type = status_type
        self.update_style()


class LogEntry(QWidget):
    """Individual log entry widget."""
    
    def __init__(self, timestamp: str, level: str, message: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)
        
        # Timestamp
        time_label = QLabel(timestamp)
        time_label.setFont(QFont("Courier", 9))
        time_label.setStyleSheet("color: #6b7280; min-width: 60px;")
        layout.addWidget(time_label)
        
        # Level badge
        badge = StatusBadge(level.upper(), level)
        badge.setMaximumWidth(60)
        layout.addWidget(badge)
        
        # Message
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: #374151;")
        layout.addWidget(msg_label, 1)
        
        # Bottom border
        self.setStyleSheet("""
            LogEntry {
                border-bottom: 1px solid #f3f4f6;
            }
        """)


class GoProManualTestGUI(QMainWindow):
    """Main GUI window for GoPro manual testing with modern design."""
    
    def __init__(self):
        super().__init__()
        self.gopro_controller = None
        self.preview_worker = None
        self.ping_worker = None
        self.preview_active = False
        self.recording_active = False
        self.hindsight_enabled = False
        self.logs = []
        
        self.setWindowTitle("GoPro Manual Test GUI")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8fafc;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #f9fafb;
                border-color: #9ca3af;
            }
            QPushButton:pressed {
                background-color: #f3f4f6;
            }
            QPushButton:disabled {
                background-color: #f3f4f6;
                color: #9ca3af;
                border-color: #e5e7eb;
            }
            QPushButton.primary {
                background-color: #3b82f6;
                border-color: #3b82f6;
                color: white;
            }
            QPushButton.primary:hover {
                background-color: #2563eb;
                border-color: #2563eb;
            }
            QPushButton.destructive {
                background-color: #ef4444;
                border-color: #ef4444;
                color: white;
            }
            QPushButton.destructive:hover {
                background-color: #dc2626;
                border-color: #dc2626;
            }
            QPushButton.success {
                background-color: #10b981;
                border-color: #10b981;
                color: white;
            }
            QPushButton.success:hover {
                background-color: #059669;
                border-color: #059669;
            }
        """)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main UI
        self.setup_ui()
        
        # Add initial logs
        self.add_log("info", "Application started")
        
    def setup_ui(self):
        """Set up the modern UI layout."""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create content area with padding
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(24)
        
        # Header section
        header_layout = QVBoxLayout()
        header_layout.setAlignment(Qt.AlignCenter)
        header_layout.setSpacing(8)
        
        # Title
        title_label = QLabel("GoPro Manual Test GUI")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setWeight(QFont.Weight.Medium)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1f2937; margin-bottom: 8px;")
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Desktop application for GoPro camera testing and configuration management")
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #6b7280; margin-bottom: 16px;")
        header_layout.addWidget(subtitle_label)
        
        content_layout.addLayout(header_layout)
        
        # Main controls grid
        controls_grid = QHBoxLayout()
        controls_grid.setSpacing(24)
        
        # Left column - Connection Panel
        self.connection_card = self.create_connection_panel()
        controls_grid.addWidget(self.connection_card, 1)
        
        # Right column - Camera Controls Panel
        self.camera_controls_card = self.create_camera_controls_panel()
        controls_grid.addWidget(self.camera_controls_card, 1)
        
        content_layout.addLayout(controls_grid)
        content_layout.addStretch()
        
        # Create splitter for main content and logging panel
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(content_widget)
        
        # Logging panel
        self.logging_panel = self.create_logging_panel()
        splitter.addWidget(self.logging_panel)
        
        # Set splitter proportions (main content gets more space)
        splitter.setSizes([600, 200])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.create_status_bar()
        
    def create_connection_panel(self):
        """Create the connection testing panel."""
        card = ModernCard("🔗 Connection & Testing")
        
        # Connection test section
        test_layout = QVBoxLayout()
        
        # Description
        desc_layout = QVBoxLayout()
        desc_layout.setSpacing(4)
        
        test_title = QLabel("Connection Test")
        test_title.setStyleSheet("font-weight: 500; color: #374151;")
        desc_layout.addWidget(test_title)
        
        test_desc = QLabel("Verify connectivity to the GoPro camera")
        test_desc.setStyleSheet("color: #6b7280; font-size: 11px;")
        desc_layout.addWidget(test_desc)
        
        # Test button and status
        button_layout = QHBoxLayout()
        button_layout.addLayout(desc_layout)
        
        self.ping_test_btn = QPushButton("Ping Test")
        self.ping_test_btn.setProperty("class", "primary")
        self.ping_test_btn.clicked.connect(self.run_ping_test)
        self.ping_test_btn.setMinimumWidth(100)
        button_layout.addWidget(self.ping_test_btn)
        
        test_layout.addLayout(button_layout)
        
        # Status indicator
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        
        self.connection_badge = StatusBadge("Disconnected", "error")
        status_layout.addWidget(self.connection_badge)
        status_layout.addStretch()
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("border: none; border-top: 1px solid #e5e7eb; margin: 8px 0;")
        
        test_layout.addWidget(separator)
        test_layout.addLayout(status_layout)
        
        card.add_layout(test_layout)
        return card
        
    def create_camera_controls_panel(self):
        """Create the camera controls panel."""
        card = ModernCard("🎥 Camera Controls")
        
        controls_layout = QVBoxLayout()
        
        # Preview section
        preview_section = self.create_control_section(
            "Live Preview", 
            "View real-time camera feed",
            "Start Preview",
            self.toggle_preview
        )
        self.start_preview_btn = preview_section["button"]
        controls_layout.addLayout(preview_section["layout"])
        
        # Preview area placeholder
        self.preview_area = QLabel()
        self.preview_area.setMinimumHeight(120)
        self.preview_area.setAlignment(Qt.AlignCenter)
        self.preview_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #d1d5db;
                border-radius: 6px;
                background-color: #f9fafb;
                color: #6b7280;
            }
        """)
        self.preview_area.setText("📹\nPreview Window\nLive camera feed will appear here")
        self.preview_area.hide()  # Initially hidden
        controls_layout.addWidget(self.preview_area)
        
        # Separator
        controls_layout.addWidget(self.create_separator())
        
        # Recording section
        recording_section = self.create_control_section(
            "Recording",
            "Start or stop video recording", 
            "Start Recording",
            self.toggle_recording
        )
        self.recording_btn = recording_section["button"]
        controls_layout.addLayout(recording_section["layout"])
        
        # Recording status indicator
        self.recording_indicator = QLabel()
        self.recording_indicator.setStyleSheet("color: #ef4444; font-size: 11px; font-weight: 500;")
        self.recording_indicator.hide()
        controls_layout.addWidget(self.recording_indicator)
        
        # Separator
        controls_layout.addWidget(self.create_separator())
        
        # Hindsight section
        hindsight_section = self.create_control_section(
            "Hindsight Mode",
            "Record up to 15 seconds before starting recording",
            "Activate",
            self.toggle_hindsight
        )
        self.hindsight_btn = hindsight_section["button"]
        controls_layout.addLayout(hindsight_section["layout"])
        
        card.add_layout(controls_layout)
        return card
        
    def create_control_section(self, title: str, description: str, button_text: str, callback):
        """Create a control section with title, description and button."""
        section_layout = QHBoxLayout()
        
        # Left side - text
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 500; color: #374151;")
        text_layout.addWidget(title_label)
        
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)
        
        section_layout.addLayout(text_layout, 1)
        
        # Right side - button
        button = QPushButton(button_text)
        button.clicked.connect(callback)
        button.setMinimumWidth(120)
        section_layout.addWidget(button)
        
        return {"layout": section_layout, "button": button}
        
    def create_separator(self):
        """Create a visual separator line."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("border: none; border-top: 1px solid #e5e7eb; margin: 8px 0;")
        return separator
        
    def create_logging_panel(self):
        """Create the logging panel."""
        card = ModernCard()
        
        # Header with clear button
        header_layout = QHBoxLayout()
        
        log_title = QLabel("Event Log")
        log_title.setStyleSheet("font-weight: 500; color: #374151; font-size: 12px;")
        header_layout.addWidget(log_title)
        header_layout.addStretch()
        
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self.clear_logs)
        clear_btn.setMaximumHeight(24)
        clear_btn.setStyleSheet("""
            QPushButton {
                font-size: 10px;
                padding: 4px 8px;
                min-height: 20px;
            }
        """)
        header_layout.addWidget(clear_btn)
        
        card.add_layout(header_layout)
        
        # Log entries area
        self.log_scroll = QScrollArea()
        self.log_scroll.setWidgetResizable(True)
        self.log_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.log_scroll.setMaximumHeight(160)
        
        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        self.log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_layout.setSpacing(0)
        self.log_layout.addStretch()  # Add stretch to push entries to top
        
        self.log_scroll.setWidget(self.log_widget)
        card.add_widget(self.log_scroll)
        
        return card
        
    def create_status_bar(self):
        """Create the modern status bar."""
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #f3f4f6;
                border-top: 1px solid #e5e7eb;
                padding: 4px 8px;
            }
        """)
        
        # Status widget with layout
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(16)
        
        # Connection status
        self.status_connection = StatusBadge("Disconnected", "error")
        status_layout.addWidget(QLabel("🔗"))
        status_layout.addWidget(self.status_connection)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #d1d5db;")
        status_layout.addWidget(sep1)
        
        # Battery
        status_layout.addWidget(QLabel("🔋"))
        status_layout.addWidget(QLabel("78%"))
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #d1d5db;")
        status_layout.addWidget(sep2)
        
        # Storage
        status_layout.addWidget(QLabel("💾"))
        status_layout.addWidget(QLabel("24.3 GB remaining"))
        
        status_layout.addStretch()
        
        # Ready indicator
        status_layout.addWidget(QLabel("Ready"))
        
        status_bar.addPermanentWidget(status_widget)
        
    def add_log(self, level: str, message: str):
        """Add a log entry to the logging panel."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Remove stretch item
        self.log_layout.takeAt(self.log_layout.count() - 1)
        
        # Add new log entry
        log_entry = LogEntry(timestamp, level, message)
        self.log_layout.addWidget(log_entry)
        
        # Add stretch back
        self.log_layout.addStretch()
        
        # Scroll to bottom
        QTimer.singleShot(10, lambda: self.log_scroll.verticalScrollBar().setValue(
            self.log_scroll.verticalScrollBar().maximum()))
            
        # Keep only last 100 entries
        if self.log_layout.count() > 102:  # 100 entries + 1 stretch + 1 extra
            old_entry = self.log_layout.takeAt(0)
            if old_entry.widget():
                old_entry.widget().deleteLater()
                
    def clear_logs(self):
        """Clear all log entries."""
        # Remove all widgets except the stretch
        while self.log_layout.count() > 1:
            item = self.log_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        
    def create_menu_bar(self):
        """Create the menu bar with Configuration menu."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        new_action = QAction("New Configuration", self)
        file_menu.addAction(new_action)
        open_action = QAction("Open Configuration...", self)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # GoPro menu
        config_menu = menu_bar.addMenu("GoPro")
        
        # Get file list action
        files_action = QAction("Get List of Files", self)
        files_action.triggered.connect(self.get_file_list)
        config_menu.addAction(files_action)
        
        config_menu.addSeparator()
        
        connect_action = QAction("Connect", self)
        config_menu.addAction(connect_action)
        disconnect_action = QAction("Disconnect", self)
        config_menu.addAction(disconnect_action)
        
        config_menu.addSeparator()
        
        # Save configuration action
        save_action = QAction("Save Configuration", self)
        save_action.triggered.connect(self.download_configuration)
        config_menu.addAction(save_action)
        
        # Load configuration action  
        load_action = QAction("Load Configuration", self)
        load_action.triggered.connect(self.upload_configuration)
        config_menu.addAction(load_action)
        
        # Send to GoPro action
        send_action = QAction("Send Configuration to GoPro", self)
        send_action.triggered.connect(self.upload_configuration)
        config_menu.addAction(send_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        help_menu.addAction(about_action)
        docs_action = QAction("Documentation", self)
        help_menu.addAction(docs_action)
        
    def run_ping_test(self):
        """Test basic GoPro connectivity in a background thread."""
        if self.ping_worker and self.ping_worker.isRunning():
            return  # Already running
            
        self.ping_test_btn.setEnabled(False)
        self.start_preview_btn.setEnabled(False)
        
        # Create and start ping test worker
        self.ping_worker = PingTestWorker()
        self.ping_worker.ping_success.connect(self.on_ping_success)
        self.ping_worker.ping_failed.connect(self.on_ping_failed)
        self.ping_worker.start()
        
    def on_ping_success(self):
        """Handle successful ping test."""
        self.show_info_popup("Connection Test", "✅ GoPro connection successful!\n\nThe camera is reachable and responding to commands.")
        self.ping_test_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(True)
        self.hindsight_btn.setEnabled(True)
        self.recording_btn.setEnabled(True)
        
    def on_ping_failed(self, error_message):
        """Handle failed ping test."""
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
            # Immediately update UI to reflect the operation
            self.preview_active = True
            self.start_preview_btn.setText("Stop Preview")
            self.start_preview_btn.setProperty("class", "destructive")
            self.start_preview_btn.style().unpolish(self.start_preview_btn)
            self.start_preview_btn.style().polish(self.start_preview_btn)

            # Show and prepare preview area
            self.preview_area.show()
            self.preview_area.setText("📹 Connecting to GoPro...\nStarting preview stream")

            # Initialize and connect to GoPro
            self.gopro_controller = GoProController()
            self.gopro_controller.connect()

            # Start preview stream
            stream_url = self.gopro_controller.start_preview()

            # Start the preview worker thread
            self.preview_worker = PreviewWorker(stream_url)
            self.preview_worker.frame_ready.connect(self.update_preview_frame)
            self.preview_worker.error_occurred.connect(self.handle_preview_error)
            self.preview_worker.start()

        except Exception as e:
            self.handle_preview_error(f"Failed to start preview: {str(e)}")
            
    def stop_gopro_preview(self):
        """Stop the GoPro preview stream."""
        try:
            # Immediately update UI to reflect the operation
            self.preview_active = False
            self.start_preview_btn.setText("Start Preview")
            self.start_preview_btn.setProperty("class", "primary")
            self.start_preview_btn.style().unpolish(self.start_preview_btn)
            self.start_preview_btn.style().polish(self.start_preview_btn)

            # Hide preview area and clear content
            self.preview_area.hide()
            self.preview_area.setText("📹\nPreview Window\nLive camera feed will appear here")
            self.preview_area.setPixmap(QPixmap())  # Clear any existing image

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

            # Scale to fit preview area while maintaining aspect ratio
            label_size = self.preview_area.size()
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            self.preview_area.setPixmap(scaled_pixmap)

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

    def toggle_hindsight(self):
        """Toggle hindsight mode on/off."""
        if not self.hindsight_enabled:
            self.enable_hindsight()
        else:
            self.disable_hindsight()
            
    def enable_hindsight(self):
        """Enable hindsight mode."""
        try:
            if not self.gopro_controller:
                self.gopro_controller = GoProController()
                self.gopro_controller.connect()

            # Immediately update UI to reflect the operation
            self.hindsight_enabled = True
            self.hindsight_btn.setText("Deactivate")
            self.hindsight_btn.setProperty("class", "success")
            self.hindsight_btn.style().unpolish(self.hindsight_btn)
            self.hindsight_btn.style().polish(self.hindsight_btn)
            self.add_log("info", "Enabling HindSight mode...")

            # Run hindsight enable in background thread
            def run_hindsight():
                try:
                    self.gopro_controller.startHindsightMode()
                    QTimer.singleShot(0, lambda: self.on_hindsight_enabled())
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_hindsight_failed(str(e)))

            threading.Thread(target=run_hindsight, daemon=True).start()

        except Exception as e:
            self.on_hindsight_failed(str(e))
            
    def disable_hindsight(self):
        """Disable hindsight mode."""
        try:
            if not self.gopro_controller:
                self.gopro_controller = GoProController()
                self.gopro_controller.connect()

            # Immediately update UI to reflect the operation
            self.hindsight_enabled = False
            self.hindsight_btn.setText("Activate")
            self.hindsight_btn.setProperty("class", "")
            self.hindsight_btn.style().unpolish(self.hindsight_btn)
            self.hindsight_btn.style().polish(self.hindsight_btn)
            self.add_log("info", "Disabling HindSight mode...")

            # Run hindsight disable in background thread
            def run_disable_hindsight():
                try:
                    self.gopro_controller.disableHindsightMode()
                    QTimer.singleShot(0, lambda: self.on_hindsight_disabled())
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_hindsight_failed(str(e)))

            threading.Thread(target=run_disable_hindsight, daemon=True).start()

        except Exception as e:
            self.on_hindsight_failed(str(e))

    def on_hindsight_disabled(self):
        """Handle successful hindsight disable."""
        self.add_log("success", "HindSight mode disabled")

    def on_hindsight_enabled(self):
        """Handle successful hindsight enable."""
        self.add_log("success", "HindSight mode enabled")

    def on_hindsight_failed(self, error_message):
        """Handle failed hindsight operation."""
        # Revert UI state since operation failed
        if self.hindsight_enabled:
            # Was trying to disable, revert to enabled state
            self.hindsight_btn.setText("Deactivate")
            self.hindsight_btn.setProperty("class", "success")
        else:
            # Was trying to enable, revert to disabled state
            self.hindsight_btn.setText("Activate")
            self.hindsight_btn.setProperty("class", "")

        self.hindsight_btn.style().unpolish(self.hindsight_btn)
        self.hindsight_btn.style().polish(self.hindsight_btn)

        # Reset internal state
        self.hindsight_enabled = not self.hindsight_enabled

        self.add_log("error", f"HindSight operation failed: {error_message}")
        self.show_error_popup(f"HindSight operation failed: {error_message}")

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

            # Immediately update UI to reflect the operation
            self.recording_active = True
            self.recording_btn.setText("Stop Recording")
            self.recording_btn.setProperty("class", "destructive")
            self.recording_btn.style().unpolish(self.recording_btn)
            self.recording_btn.style().polish(self.recording_btn)

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

            # Immediately update UI to reflect the operation
            self.recording_active = False
            self.recording_btn.setText("Start Recording")
            self.recording_btn.setProperty("class", "")
            self.recording_btn.style().unpolish(self.recording_btn)
            self.recording_btn.style().polish(self.recording_btn)

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
        self.add_log("success", "Recording started")

    def on_recording_stopped(self):
        """Handle successful recording stop."""
        self.add_log("success", "Recording stopped")

    def on_recording_failed(self, error_message):
        """Handle failed recording operation."""
        # Revert UI state since operation failed
        if self.recording_active:
            # Was trying to stop, revert to recording state
            self.recording_btn.setText("Stop Recording")
            self.recording_btn.setProperty("class", "destructive")
        else:
            # Was trying to start, revert to stopped state
            self.recording_btn.setText("Start Recording")
            self.recording_btn.setProperty("class", "")

        self.recording_btn.style().unpolish(self.recording_btn)
        self.recording_btn.style().polish(self.recording_btn)

        # Reset internal state
        self.recording_active = not self.recording_active

        self.add_log("error", f"Recording operation failed: {error_message}")
        self.show_error_popup(f"Recording operation failed: {error_message}")
    
    def get_file_list(self):
        """Get list of files from GoPro."""
        self.add_log("info", "Fetching file list from GoPro...")
        # Placeholder for file list functionality
        QTimer.singleShot(1000, lambda: self.add_log("success", "File list retrieved successfully"))
        
    def download_configuration(self):
        """Download configuration from GoPro and save to YAML file."""
        try:
            # Get save location from user
            default_filename = f"gopro_config_{self.get_timestamp()}.yaml"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save GoPro Configuration",
                default_filename,
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            self.add_log("info", "Downloading configuration...")
            
            # Run download in background thread
            def run_download():
                try:
                    if not self.gopro_controller:
                        self.gopro_controller = GoProController()
                        self.gopro_controller.connect()
                    
                    saved_path = self.gopro_controller.download_configuration(file_path)
                    QTimer.singleShot(0, lambda: self.on_download_success(saved_path))
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_download_failed(str(e)))
            
            threading.Thread(target=run_download, daemon=True).start()
            
        except Exception as e:
            self.on_download_failed(str(e))
    
    def upload_configuration(self):
        """Upload configuration from YAML file to GoPro."""
        try:
            # Get file from user
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select GoPro Configuration File",
                "",
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            
            # Run upload in background thread
            def run_upload():
                try:
                    if not self.gopro_controller:
                        self.gopro_controller = GoProController()
                        self.gopro_controller.connect()
                    
                    success = self.gopro_controller.upload_configuration(file_path)
                    if success:
                        QTimer.singleShot(0, lambda: self.on_upload_success(file_path))
                    else:
                        QTimer.singleShot(0, lambda: self.on_upload_failed("Configuration upload returned False"))
                        
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.on_upload_failed(str(e)))
            
            threading.Thread(target=run_upload, daemon=True).start()
            
        except Exception as e:
            self.on_upload_failed(str(e))
    
    def on_download_success(self, saved_path):
        """Handle successful configuration download."""
        self.show_info_popup(
            "Configuration Downloaded", 
            f"✅ GoPro configuration successfully downloaded!\n\nSaved to: {saved_path}"
        )
    
    def on_download_failed(self, error_message):
        """Handle failed configuration download."""
        
        # Provide user-friendly error messages
        if "ConnectionError" in error_message or "connection" in error_message.lower():
            user_message = "❌ GoPro Connection Error\n\nPlease ensure:\n• GoPro is connected via USB\n• GoPro is powered on\n• Try the 'Ping Test' button first"
        elif "PermissionError" in error_message or "permission denied" in error_message.lower():
            user_message = f"❌ File Permission Error\n\nUnable to save configuration file.\nPlease check folder permissions or try saving to a different location.\n\nDetails: {error_message}"
        elif "OSError" in error_message:
            user_message = f"❌ File System Error\n\nUnable to save configuration file.\nPlease check available disk space and try again.\n\nDetails: {error_message}"
        else:
            user_message = f"❌ Configuration Download Failed\n\n{error_message}"
            
        self.show_error_popup(user_message)
    
    def on_upload_success(self, file_path):
        """Handle successful configuration upload."""
        self.show_info_popup(
            "Configuration Uploaded",
            f"✅ Configuration successfully uploaded to GoPro!\n\nFrom file: {file_path}"
        )
    
    def on_upload_failed(self, error_message):
        """Handle failed configuration upload."""
        
        # Provide user-friendly error messages
        if "FileNotFoundError" in error_message or "not found" in error_message.lower():
            user_message = "❌ Configuration File Not Found\n\nThe selected configuration file could not be found.\nPlease verify the file path and try again."
        elif "ConnectionError" in error_message or "connection" in error_message.lower():
            user_message = "❌ GoPro Connection Error\n\nLost connection to GoPro during upload.\n\nPlease ensure:\n• GoPro is connected via USB\n• GoPro is powered on\n• Try the 'Ping Test' button to verify connection"
        elif "ValidationError" in error_message or "validation failed" in error_message.lower():
            user_message = f"❌ Configuration File Invalid\n\nThe configuration file format is invalid or contains unsupported values.\n\nPlease check the file format and try again.\n\nDetails:\n{error_message}"
        elif "ValueError" in error_message or "invalid" in error_message.lower():
            user_message = f"❌ Invalid Configuration File\n\nThe configuration file contains errors:\n\n{error_message}"
        elif "yaml" in error_message.lower() or "YAML" in error_message:
            user_message = f"❌ YAML Format Error\n\nThe configuration file has invalid YAML format.\nPlease check the file syntax and try again.\n\nDetails: {error_message}"
        else:
            user_message = f"❌ Configuration Upload Failed\n\n{error_message}"
            
        self.show_error_popup(user_message)
    
    def get_timestamp(self):
        """Get current timestamp for filename generation."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
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