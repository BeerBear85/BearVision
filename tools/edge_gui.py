"""
EDGE Application GUI

A PySide6 GUI application for manually interacting with the BearVision EDGE device.
Provides real-time monitoring of detection status, video preview, and event logging.
Based on the React mock-up design from temp/EDGE Application GUI Design/.

Features:
- Real-time video preview with YOLO detection overlays
- Status indicators for system state (Active, Hindsight Mode, Recording, Preview)
- Event log with timestamped messages
- Dark theme interface
- Manual controls for EDGE device functionality
"""

import sys
import os
import threading
import cv2
import numpy as np
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QTextEdit, QScrollArea, QGridLayout,
    QGroupBox, QSplitter, QListWidget, QListWidgetItem, QSizePolicy,
    QProgressBar, QMenuBar, QFileDialog, QMessageBox, QSpacerItem
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QAction, QIcon, QColor, QPalette, QPainter, QPen

# Add module paths
MODULE_DIR = Path(__file__).resolve().parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

# Import EDGE application modules
try:
    from ConfigurationHandler import ConfigurationHandler
    from GoProController import GoProController
    import edge_main
    from edge_application import EdgeApplication, EdgeStatus, SystemStatus, DetectionResult
    EDGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"EDGE modules not available: {e}")
    EDGE_AVAILABLE = False


class EventType(Enum):
    """Event types for logging."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class DetectionBox:
    """YOLO detection bounding box."""
    id: str
    x: float  # Percentage
    y: float  # Percentage
    width: float  # Percentage
    height: float  # Percentage
    label: str
    confidence: float = 0.0


@dataclass
class Event:
    """Event log entry."""
    id: str
    timestamp: str
    type: EventType
    message: str


@dataclass
class StatusIndicators:
    """System status indicators."""
    active: bool = False
    hindsight_mode: bool = False
    recording: bool = False
    preview: bool = False


class EDGEBackend(QThread):
    """Backend thread for EDGE application integration using EdgeApplication."""

    # Signals for communicating with GUI
    motion_detected = Signal()
    hindsight_triggered = Signal()
    status_changed = Signal(StatusIndicators)
    status_message_changed = Signal(str)
    log_event = Signal(EventType, str)
    preview_frame = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.edge_app: Optional[EdgeApplication] = None
        self.running = False
        self.status = StatusIndicators()

        # Initialize EdgeApplication with callbacks
        if EDGE_AVAILABLE:
            self.edge_app = EdgeApplication(
                status_callback=self._on_status_update,
                detection_callback=self._on_detection,
                ble_callback=self._on_ble_data,
                log_callback=self._on_log,
                frame_callback=self._on_frame_received
            )

    def _on_status_update(self, edge_status: SystemStatus):
        """Handle status updates from EdgeApplication."""
        # Convert EdgeApplication status to GUI status
        self.status.active = edge_status.gopro_connected
        self.status.preview = edge_status.preview_active
        self.status.recording = edge_status.recording
        self.status.hindsight_mode = edge_status.hindsight_mode

        # Map EdgeStatus to status bar messages
        status_messages = {
            EdgeStatus.INITIALIZING: "Initializing EDGE system...",
            EdgeStatus.READY: "EDGE system ready",
            EdgeStatus.ACTIVE: "EDGE system active",
            EdgeStatus.LOOKING_FOR_WAKEBOARDER: "Looking for wakeboarder",
            EdgeStatus.MOTION_DETECTED: "Motion detected - analyzing...",
            EdgeStatus.RECORDING: "Recording highlight clip",
            EdgeStatus.ERROR: "System error",
            EdgeStatus.STOPPED: "System stopped"
        }

        status_message = status_messages.get(edge_status.overall_status, "Unknown status")
        self.status_message_changed.emit(status_message)
        self.log_event.emit(EventType.INFO, f"Status: {status_message}")

        self.status_changed.emit(self.status)

    def _on_detection(self, detection: DetectionResult):
        """Handle detection results from EdgeApplication."""
        self.motion_detected.emit()

    def _on_ble_data(self, ble_data: Dict[str, Any]):
        """Handle BLE data from EdgeApplication."""
        # Log BLE data for now
        acc = ble_data.get('acc_sensor')
        if acc:
            self._on_log("debug", f"BLE: {acc.get_value_string()}")

    def _on_log(self, level: str, message: str):
        """Handle log messages from EdgeApplication."""
        level_map = {
            "info": EventType.INFO,
            "warning": EventType.WARNING,
            "error": EventType.ERROR,
            "debug": EventType.INFO
        }
        event_type = level_map.get(level, EventType.INFO)
        self.log_event.emit(event_type, message)

    def _on_frame_received(self, frame: np.ndarray):
        """Handle preview frames from EdgeApplication."""
        # Emit the frame to the GUI
        self.preview_frame.emit(frame)

    def initialize_edge(self):
        """Initialize EDGE system."""
        if not EDGE_AVAILABLE or not self.edge_app:
            self.log_event.emit(EventType.ERROR, "EDGE modules not available")
            return False

        return self.edge_app.initialize()

    def connect_gopro(self):
        """Connect to GoPro camera."""
        if not self.edge_app:
            return False

        return self.edge_app.connect_gopro()

    def start_preview(self):
        """Start GoPro preview."""
        if not self.edge_app:
            return False

        return self.edge_app.start_preview()

    def stop_preview(self):
        """Stop GoPro preview."""
        if not self.edge_app:
            return False

        return self.edge_app.stop_preview()

    def start_edge_processing(self):
        """Start complete EDGE system."""
        if not self.edge_app:
            return False

        success = self.edge_app.start_system()
        if success:
            self.running = True
            self.log_event.emit(EventType.SUCCESS, "EDGE system started - Looking for wakeboarder")

        return success

    def trigger_hindsight(self):
        """Manually trigger hindsight clip."""
        if not self.edge_app:
            return False

        success = self.edge_app.trigger_hindsight()
        if success:
            self.hindsight_triggered.emit()

        return success

    def stop_edge(self):
        """Stop EDGE processing."""
        self.running = False
        if self.edge_app:
            self.edge_app.stop_system()

    def run(self):
        """Main backend thread loop."""
        # Keep the thread alive and handle any background processing
        while self.running:
            self.msleep(1000)  # Sleep for 1 second


class StatusBar(QWidget):
    """Top status bar with logo and dynamic status message."""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup the status bar UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)

        # Left side - Logo and title
        left_layout = QHBoxLayout()

        # Load and display the actual BearVision logo
        logo_label = QLabel()
        try:
            logo_path = Path(__file__).resolve().parent.parent / "logo" / "Logo.png"
            logo_pixmap = QPixmap(str(logo_path))
            if not logo_pixmap.isNull():
                # Scale logo to appropriate size
                scaled_logo = logo_pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled_logo)
            else:
                logo_label.setText("🐻")  # Fallback to emoji
                logo_label.setStyleSheet("font-size: 32px;")
        except Exception as e:
            logging.warning(f"Failed to load logo: {e}")
            logo_label.setText("🐻")  # Fallback to emoji
            logo_label.setStyleSheet("font-size: 32px;")

        left_layout.addWidget(logo_label)

        title_label = QLabel("EDGE Application")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white; margin-left: 10px;")
        left_layout.addWidget(title_label)

        # Right side - Status message
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            background-color: #44403c;
            color: white;
            padding: 6px 12px;
            border: 1px solid #57534e;
            font-weight: 500;
            font-size: 12px;
        """)

        layout.addLayout(left_layout)
        layout.addStretch()
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #252525; border-bottom: 1px solid #404040;")

    def update_status(self, message: str):
        """Update the status message."""
        self.status_label.setText(message)


class PreviewArea(QWidget):
    """Video preview area with YOLO detection overlays."""

    def __init__(self):
        super().__init__()
        self.detections: List[DetectionBox] = []
        self.current_image: Optional[QImage] = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the preview area UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Preview header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            background-color: #252525;
            border-bottom: 1px solid #404040;
            padding: 8px 12px;
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 8, 12, 8)

        header_label = QLabel("Video Preview")
        header_label.setStyleSheet("color: white; font-weight: 500; font-size: 14px;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        layout.addWidget(header_frame)

        # Main preview area
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #404040;
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No preview available")
        self.image_label.setScaledContents(True)

        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.setStyleSheet("""
            background-color: #171717;
            border: 1px solid #404040;
        """)

    def update_image(self, image: np.ndarray):
        """Update the preview image."""
        if image is None:
            return

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width

        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.current_image = q_image

        # Draw detection boxes
        self.draw_detections()

    def draw_detections(self):
        """Draw YOLO detection boxes on the image."""
        if self.current_image is None:
            return

        pixmap = QPixmap.fromImage(self.current_image)

        if self.detections:
            painter = QPainter(pixmap)
            pen = QPen(QColor(34, 197, 94), 3)  # Green color
            painter.setPen(pen)
            painter.setFont(QFont("Arial", 12, QFont.Bold))

            img_width = pixmap.width()
            img_height = pixmap.height()

            for detection in self.detections:
                # Convert percentage to actual coordinates
                x = int(detection.x * img_width / 100)
                y = int(detection.y * img_height / 100)
                w = int(detection.width * img_width / 100)
                h = int(detection.height * img_height / 100)

                # Draw bounding box
                painter.drawRect(x, y, w, h)

                # Draw label
                label_text = f"{detection.label}"
                if detection.confidence > 0:
                    label_text += f" ({detection.confidence:.2f})"

                label_rect = painter.fontMetrics().boundingRect(label_text)
                label_bg_rect = label_rect.adjusted(-4, -2, 4, 2)
                label_bg_rect.translate(x, y - label_rect.height() - 5)

                painter.fillRect(label_bg_rect, QColor(34, 197, 94))
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(label_bg_rect, Qt.AlignCenter, label_text)
                painter.setPen(pen)

            painter.end()

        self.image_label.setPixmap(pixmap)

    def update_detections(self, detections: List[DetectionBox]):
        """Update detection boxes."""
        self.detections = detections
        self.draw_detections()


class IndicatorWidget(QWidget):
    """Individual status indicator widget."""

    def __init__(self, label: str, icon: str, active: bool = False):
        super().__init__()
        self.label = label
        self.icon = icon
        self.active = active
        self.setup_ui()

    def setup_ui(self):
        """Setup the indicator UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Icon
        self.icon_label = QLabel(self.icon)
        self.icon_label.setFixedSize(24, 24)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.update_icon_style()

        # Label and status
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        self.label_text = QLabel(self.label)
        self.label_text.setStyleSheet("color: white; font-weight: 500; font-size: 11px;")

        self.status_text = QLabel("OFF")
        self.update_status_style()

        text_layout.addWidget(self.label_text)
        text_layout.addWidget(self.status_text)

        layout.addWidget(self.icon_label)
        layout.addLayout(text_layout)
        layout.addStretch()

        self.setLayout(layout)

    def update_icon_style(self):
        """Update icon styling based on active state."""
        bg_color = "#22c55e" if self.active else "#ef4444"  # Green or Red
        self.icon_label.setStyleSheet(f"""
            background-color: {bg_color};
            color: white;
            border: 1px solid #404040;
            font-size: 12px;
            font-weight: bold;
        """)

    def update_status_style(self):
        """Update status text styling."""
        color = "#22c55e" if self.active else "#ef4444"  # Green or Red
        text = "ON" if self.active else "OFF"
        self.status_text.setText(text)
        self.status_text.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")

    def set_active(self, active: bool):
        """Set the active state."""
        self.active = active
        self.update_icon_style()
        self.update_status_style()


class IndicatorsPanel(QWidget):
    """Status indicators panel."""

    def __init__(self):
        super().__init__()
        self.indicators: Dict[str, IndicatorWidget] = {}
        self.setup_ui()

    def setup_ui(self):
        """Setup the indicators panel UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            background-color: #252525;
            border-bottom: 1px solid #404040;
            padding: 8px 12px;
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("Status Indicators")
        title.setStyleSheet("color: white; font-weight: 500; font-size: 14px;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        layout.addWidget(header_frame)

        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)

        # Create indicators
        indicators_data = [
            ("active", "Active", "⚡"),
            ("hindsight", "Hindsight Mode", "👁"),
            ("recording", "Recording", "🎥"),
            ("preview", "Preview", "🖥")
        ]

        for key, label, icon in indicators_data:
            indicator = IndicatorWidget(label, icon)
            self.indicators[key] = indicator
            content_layout.addWidget(indicator)

        content_layout.addStretch()
        layout.addWidget(content_widget)

        self.setLayout(layout)
        self.setStyleSheet("""
            background-color: #252525;
            border: 1px solid #404040;
        """)

    def update_indicators(self, status: StatusIndicators):
        """Update all indicators."""
        self.indicators["active"].set_active(status.active)
        self.indicators["hindsight"].set_active(status.hindsight_mode)
        self.indicators["recording"].set_active(status.recording)
        self.indicators["preview"].set_active(status.preview)


class EventList(QWidget):
    """Event log list widget."""

    def __init__(self):
        super().__init__()
        self.events: List[Event] = []
        self.setup_ui()

    def setup_ui(self):
        """Setup the event list UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            background-color: #252525;
            border-bottom: 1px solid #404040;
            padding: 8px 12px;
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("Event Log")
        title.setStyleSheet("color: white; font-weight: 500; font-size: 14px;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        layout.addWidget(header_frame)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("""
            background-color: #252525;
            color: white;
            border: none;
            font-family: 'Courier New', monospace;
            font-size: 10px;
            padding: 8px;
        """)
        layout.addWidget(self.text_area)

        self.setLayout(layout)
        self.setStyleSheet("""
            background-color: #252525;
            border: 1px solid #404040;
        """)

    def add_event(self, event: Event):
        """Add a new event to the log."""
        self.events.append(event)

        # Color mapping
        color_map = {
            EventType.INFO: "#60a5fa",      # Blue
            EventType.WARNING: "#fbbf24",   # Yellow
            EventType.ERROR: "#f87171",     # Red
            EventType.SUCCESS: "#34d399"    # Green
        }

        type_labels = {
            EventType.INFO: "[INFO]",
            EventType.WARNING: "[WARN]",
            EventType.ERROR: "[ERROR]",
            EventType.SUCCESS: "[SUCCESS]"
        }

        color = color_map.get(event.type, "#60a5fa")
        type_label = type_labels.get(event.type, "[INFO]")

        # Format event line with better spacing like the mockup
        event_html = f'<span style="color: #9ca3af; font-size: 10px;">{event.timestamp}</span>&nbsp;&nbsp;<span style="color: {color}; font-size: 10px;">{type_label}</span>&nbsp;&nbsp;<span style="color: white; font-size: 10px;">{event.message}</span>'

        self.text_area.append(event_html)

        # Scroll to bottom
        scrollbar = self.text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class EDGEMainWindow(QMainWindow):
    """Main EDGE application window."""

    def __init__(self):
        super().__init__()
        self.status_indicators = StatusIndicators()
        self.backend = EDGEBackend()
        self.demo_active = False  # Track demo mode state
        self.setup_ui()
        self.setup_backend_connections()
        self.setup_demo_data()

        # Start backend thread
        self.backend.start()

    def setup_ui(self):
        """Setup the main window UI."""
        self.setWindowTitle("BearVision EDGE Application")
        self.setMinimumSize(1200, 800)

        # Set dark theme to match mockup
        self.setStyleSheet("""
            QMainWindow {
                background-color: #252525;
                color: white;
            }
        """)

        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Status bar with fixed height (like mockup)
        self.status_bar = StatusBar()
        self.status_bar.setFixedHeight(64)  # Fixed height like in mockup
        main_layout.addWidget(self.status_bar)

        # Content area with 4-column grid layout (3:1 ratio like mockup)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)

        # Left side - Preview area (takes 3/4 of space)
        self.preview_area = PreviewArea()
        content_layout.addWidget(self.preview_area, 3)

        # Right side - Indicators and events (takes 1/4 of space)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(4)

        self.indicators_panel = IndicatorsPanel()
        right_layout.addWidget(self.indicators_panel, 1)

        self.event_list = EventList()
        right_layout.addWidget(self.event_list, 1)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        content_layout.addWidget(right_widget, 1)

        main_layout.addLayout(content_layout)
        main_widget.setLayout(main_layout)

        # Setup menu bar
        self.setup_menu_bar()

    def setup_backend_connections(self):
        """Setup connections between backend and GUI."""
        # Connect backend signals to GUI updates
        self.backend.log_event.connect(self.handle_backend_log_event)
        self.backend.status_changed.connect(self.handle_status_changed)
        self.backend.status_message_changed.connect(self.handle_status_message_changed)
        self.backend.motion_detected.connect(self.handle_motion_detected)
        self.backend.hindsight_triggered.connect(self.handle_hindsight_triggered)
        self.backend.preview_frame.connect(self.handle_preview_frame)

    def handle_backend_log_event(self, event_type: EventType, message: str):
        """Handle log events from backend."""
        self.add_log_event(event_type, message)

    def handle_status_changed(self, status: StatusIndicators):
        """Handle status changes from backend."""
        self.status_indicators = status
        self.indicators_panel.update_indicators(status)

    def handle_status_message_changed(self, message: str):
        """Handle status message changes from backend."""
        self.status_bar.update_status(message)

    def handle_motion_detected(self):
        """Handle motion detection from backend."""
        self.status_bar.update_status("Motion detected - analyzing...")

    def handle_hindsight_triggered(self):
        """Handle hindsight trigger from backend."""
        self.status_bar.update_status("Hindsight clip triggered")

    def handle_preview_frame(self, frame: np.ndarray):
        """Handle preview frame from backend."""
        self.preview_area.update_image(frame)

    def setup_menu_bar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        init_action = QAction("Initialize EDGE System", self)
        init_action.triggered.connect(self.initialize_edge_system)
        tools_menu.addAction(init_action)

        connect_action = QAction("Connect to GoPro", self)
        connect_action.triggered.connect(self.connect_gopro)
        tools_menu.addAction(connect_action)

        start_preview_action = QAction("Start Preview", self)
        start_preview_action.triggered.connect(self.start_preview)
        tools_menu.addAction(start_preview_action)

        stop_preview_action = QAction("Stop Preview", self)
        stop_preview_action.triggered.connect(self.stop_preview)
        tools_menu.addAction(stop_preview_action)

        tools_menu.addSeparator()

        start_edge_action = QAction("Start EDGE Processing", self)
        start_edge_action.triggered.connect(self.start_edge_processing)
        tools_menu.addAction(start_edge_action)

        trigger_hindsight_action = QAction("Trigger Hindsight Manually", self)
        trigger_hindsight_action.triggered.connect(self.trigger_hindsight)
        tools_menu.addAction(trigger_hindsight_action)

        stop_edge_action = QAction("Stop EDGE Processing", self)
        stop_edge_action.triggered.connect(self.stop_edge_processing)
        tools_menu.addAction(stop_edge_action)

        # Demo menu
        demo_menu = menubar.addMenu("Demo")

        start_demo_action = QAction("Start Demo Mode", self)
        start_demo_action.triggered.connect(self.start_demo_mode)
        demo_menu.addAction(start_demo_action)

        stop_demo_action = QAction("Stop Demo Mode", self)
        stop_demo_action.triggered.connect(self.stop_demo_mode)
        demo_menu.addAction(stop_demo_action)

    def setup_demo_data(self):
        """Setup demo data for testing."""
        # Add some demo events
        demo_events = [
            Event("1", "14:32:15", EventType.SUCCESS, "Wakeboarder detected successfully"),
            Event("2", "14:32:12", EventType.INFO, "YOLO model inference completed"),
            Event("3", "14:32:10", EventType.INFO, "Processing frame 2847"),
            Event("4", "14:32:08", EventType.WARNING, "Low confidence detection threshold"),
            Event("5", "14:32:05", EventType.INFO, "Camera feed initialized"),
            Event("6", "14:32:03", EventType.SUCCESS, "Application started successfully")
        ]

        for event in demo_events:
            self.event_list.add_event(event)

        # Setup demo detections
        self.demo_detections = [
            DetectionBox("person-1", 25, 30, 15, 25, "Wakeboarder", 0.85),
            DetectionBox("person-2", 65, 45, 12, 20, "Person", 0.72)
        ]

        # Update indicators
        self.status_indicators.active = True
        self.status_indicators.preview = True
        self.status_indicators.recording = True
        self.indicators_panel.update_indicators(self.status_indicators)

    def start_demo_simulation(self):
        """Start demo simulation with timers."""
        # Status message rotation timer
        self.status_messages = [
            "Searching for wakeboarder...",
            "Processing video feed...",
            "Analyzing motion patterns...",
            "Wakeboarder detected - tracking...",
            "Recording highlight reel..."
        ]
        self.status_index = 0

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_demo_status)
        self.status_timer.start(3000)  # 3 seconds

        # Indicators update timer
        self.indicators_timer = QTimer()
        self.indicators_timer.timeout.connect(self.update_demo_indicators)
        self.indicators_timer.start(5000)  # 5 seconds

    def update_demo_status(self):
        """Update demo status message."""
        message = self.status_messages[self.status_index]
        self.status_bar.update_status(message)
        self.status_index = (self.status_index + 1) % len(self.status_messages)

    def update_demo_indicators(self):
        """Update demo indicators."""
        self.status_indicators.hindsight_mode = not self.status_indicators.hindsight_mode
        self.indicators_panel.update_indicators(self.status_indicators)

    def start_demo_mode(self):
        """Start demo mode manually."""
        if not self.demo_active:
            self.demo_active = True
            self.start_demo_simulation()
            self.add_log_event(EventType.INFO, "Demo mode started")

    def stop_demo_mode(self):
        """Stop demo mode manually."""
        if self.demo_active:
            self.demo_active = False
            # Stop demo timers
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()
            if hasattr(self, 'indicators_timer'):
                self.indicators_timer.stop()
            self.add_log_event(EventType.INFO, "Demo mode stopped")
            self.status_bar.update_status("Demo mode stopped")

    def initialize_edge_system(self):
        """Initialize EDGE system."""
        self.add_log_event(EventType.INFO, "Initializing EDGE system...")
        success = self.backend.initialize_edge()
        if success:
            self.status_bar.update_status("EDGE system initialized")
        else:
            self.status_bar.update_status("EDGE system initialization failed")

    def connect_gopro(self):
        """Connect to GoPro camera."""
        self.add_log_event(EventType.INFO, "Connecting to GoPro...")
        success = self.backend.connect_gopro()
        if success:
            self.status_bar.update_status("GoPro connected")
        else:
            self.status_bar.update_status("GoPro connection failed")

    def start_preview(self):
        """Start video preview."""
        self.add_log_event(EventType.INFO, "Starting preview...")
        success = self.backend.start_preview()
        if success:
            self.status_bar.update_status("Preview active")
            # Preview frames will come from the backend via preview_frame signal
        else:
            self.status_bar.update_status("Preview start failed")

    def stop_preview(self):
        """Stop video preview."""
        self.add_log_event(EventType.INFO, "Stopping preview...")
        success = self.backend.stop_preview()
        if success:
            self.status_bar.update_status("Preview stopped")
        else:
            self.status_bar.update_status("Preview stop failed")

    def start_edge_processing(self):
        """Start EDGE processing."""
        self.add_log_event(EventType.INFO, "Starting EDGE processing...")
        success = self.backend.start_edge_processing()
        if success:
            self.status_bar.update_status("EDGE processing active")
        else:
            self.status_bar.update_status("EDGE processing start failed")

    def trigger_hindsight(self):
        """Trigger hindsight clip manually."""
        self.add_log_event(EventType.INFO, "Triggering hindsight clip...")
        self.backend.trigger_hindsight()

    def stop_edge_processing(self):
        """Stop EDGE processing."""
        self.add_log_event(EventType.INFO, "Stopping EDGE processing...")
        self.backend.stop_edge()
        self.status_bar.update_status("EDGE processing stopped")

    def add_log_event(self, event_type: EventType, message: str):
        """Add a new event to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        event_id = str(len(self.event_list.events) + 1)
        event = Event(event_id, timestamp, event_type, message)
        self.event_list.add_event(event)

    def closeEvent(self, event):
        """Handle application close event."""
        self.add_log_event(EventType.INFO, "Shutting down EDGE application...")

        # Stop backend processing
        self.backend.stop_edge()

        # Wait for backend thread to finish
        if self.backend.isRunning():
            self.backend.quit()
            self.backend.wait(3000)  # Wait up to 3 seconds

        # Stop demo timers
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        if hasattr(self, 'indicators_timer'):
            self.indicators_timer.stop()

        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("BearVision EDGE GUI")
    app.setApplicationVersion("1.0.0")

    # Create and show main window
    window = EDGEMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()