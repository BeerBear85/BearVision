"""
BLE Manual Test GUI

A PySide6-based GUI application for manual testing of BLE tags.
Provides start/stop logging controls, real-time data display in a scrollable window,
and YAML-based configuration.
"""

import sys
import os
import asyncio
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFrame, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QTextCursor

# Add module path for BLE handler
MODULE_DIR = Path(__file__).resolve().parent / "code" / "modules"
sys.path.append(str(MODULE_DIR))

from ble_beacon_handler import BleBeaconHandler, AccSensorValue


class BleDataSignals(QObject):
    """Signals for thread-safe BLE data updates."""
    data_received = Signal(dict)
    error_occurred = Signal(str)
    logging_started = Signal()
    logging_stopped = Signal()


class BleWorker(QThread):
    """Worker thread for BLE scanning and data collection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.signals = BleDataSignals()
        self.ble_handler = BleBeaconHandler()
        self.running = False
        self.data_count = 0
        
    def run(self):
        """Start BLE scanning in worker thread."""
        try:
            self.running = True
            self.signals.logging_started.emit()
            
            # Run the BLE scanning with custom processing
            asyncio.run(self._scan_and_process())
            
        except Exception as e:
            self.signals.error_occurred.emit(f"BLE scanning error: {str(e)}")
        finally:
            self.running = False
            self.signals.logging_stopped.emit()
            
    async def _scan_and_process(self):
        """Custom BLE scanning and processing."""
        # Start scanning task
        scan_timeout = self.config.get('ble', {}).get('scan_timeout', 0.0)
        scan_task = asyncio.create_task(
            self.ble_handler.look_for_advertisements(scan_timeout)
        )
        
        # Process advertisements as they arrive
        process_task = asyncio.create_task(self._process_advertisements())
        
        # Wait for either scan completion or stop signal
        await asyncio.gather(scan_task, process_task, return_exceptions=True)
        
    async def _process_advertisements(self):
        """Process BLE advertisements and emit data signals."""
        while self.running:
            try:
                # Get advertisement data with timeout
                advertisement = await asyncio.wait_for(
                    self.ble_handler.advertisement_queue.get(), 
                    timeout=1.0
                )
                
                self.data_count += 1
                advertisement['data_count'] = self.data_count
                advertisement['timestamp'] = datetime.now()
                
                # Emit signal for GUI update
                self.signals.data_received.emit(advertisement)
                
                # Mark task as done
                self.ble_handler.advertisement_queue.task_done()
                
            except asyncio.TimeoutError:
                # Normal timeout, continue if still running
                continue
            except Exception as e:
                if self.running:  # Only report errors if we're still supposed to be running
                    self.signals.error_occurred.emit(f"Data processing error: {str(e)}")
                break
                
    def stop(self):
        """Stop BLE scanning."""
        self.running = False
        self.quit()
        self.wait()


class BleManualTestGUI(QMainWindow):
    """Main GUI window for BLE manual testing."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize state
        self.config = {}
        self.ble_worker = None
        self.logging_active = False
        self.data_count = 0
        
        # Load configuration
        self.load_configuration()
        
        # Setup logging
        self.setup_logging()
        
        # Create GUI
        self.setup_ui()
        
    def load_configuration(self):
        """Load YAML configuration file."""
        config_file = Path(__file__).parent / "ble_test_gui_config.yaml"
        
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                
        except FileNotFoundError:
            self.show_error_popup(
                "Configuration file not found",
                f"Could not find config file: {config_file}\n"
                "Using default settings."
            )
            self.config = self.get_default_config()
            
        except yaml.YAMLError as e:
            self.show_error_popup(
                "Invalid YAML configuration",
                f"Error parsing config file: {str(e)}\n"
                "Using default settings."
            )
            self.config = self.get_default_config()
            
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails."""
        return {
            'ble': {
                'scan_timeout': 0.0,
                'movement_threshold': 0.1,
            },
            'gui': {
                'window_title': 'BLE Tag Manual Test GUI',
                'window_width': 800,
                'window_height': 600,
                'log_max_lines': 1000,
                'log_font_family': 'Consolas',
                'log_font_size': 9,
                'auto_scroll': True,
                'show_timestamp': True,
                'show_address': True,
                'show_name': True,
                'show_rssi': True,
                'show_distance': True,
                'show_battery': True,
                'show_accelerometer': True,
                'show_movement': True,
            }
        }
        
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
        
    def setup_ui(self):
        """Create and setup the user interface."""
        gui_config = self.config.get('gui', {})
        
        # Window configuration
        self.setWindowTitle(gui_config.get('window_title', 'BLE Tag Manual Test GUI'))
        self.setGeometry(
            100, 100,
            gui_config.get('window_width', 800),
            gui_config.get('window_height', 600)
        )
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("BLE Tag Manual Testing")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Control panel
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        
        # Start/Stop logging buttons
        self.start_button = QPushButton("Start Logging")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.start_logging)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Logging") 
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_logging)
        control_layout.addWidget(self.stop_button)
        
        # Status display
        self.status_label = QLabel("Ready - Click 'Start Logging' to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_frame)
        
        # Data counter
        self.counter_label = QLabel("Data packets received: 0")
        self.counter_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.counter_label)
        
        # Scrollable log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        
        # Configure log font
        log_font = QFont(
            gui_config.get('log_font_family', 'Consolas'),
            gui_config.get('log_font_size', 9)
        )
        self.log_display.setFont(log_font)
        
        # Set background and text colors if configured
        bg_color = gui_config.get('background_color')
        text_color = gui_config.get('text_color')
        if bg_color or text_color:
            style = "QTextEdit {"
            if bg_color:
                style += f"background-color: {bg_color};"
            if text_color:
                style += f"color: {text_color};"
            style += "}"
            self.log_display.setStyleSheet(style)
            
        main_layout.addWidget(self.log_display)
        
        # Instructions
        instructions = QLabel(
            "Instructions: Click 'Start Logging' to begin scanning for KBPro BLE tags. "
            "Real-time data will appear above. Click 'Stop Logging' to stop."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666666; font-size: 10px; padding: 5px;")
        main_layout.addWidget(instructions)
        
    def start_logging(self):
        """Start BLE data logging."""
        try:
            if self.ble_worker is not None and self.ble_worker.isRunning():
                return  # Already running
                
            self.logger.info("Starting BLE logging...")
            
            # Create and start BLE worker thread
            self.ble_worker = BleWorker(self.config)
            self.ble_worker.signals.data_received.connect(self.handle_ble_data)
            self.ble_worker.signals.error_occurred.connect(self.handle_ble_error)
            self.ble_worker.signals.logging_started.connect(self.on_logging_started)
            self.ble_worker.signals.logging_stopped.connect(self.on_logging_stopped)
            
            self.ble_worker.start()
            
        except Exception as e:
            self.show_error_popup("Failed to start logging", str(e))
            
    def stop_logging(self):
        """Stop BLE data logging."""
        try:
            if self.ble_worker is not None:
                self.logger.info("Stopping BLE logging...")
                self.ble_worker.stop()
                
        except Exception as e:
            self.show_error_popup("Error stopping logging", str(e))
            
    def on_logging_started(self):
        """Handle logging started event."""
        self.logging_active = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Logging active - Scanning for BLE tags...")
        
        # Add startup message to log
        self.add_log_message("=== BLE Logging Started ===")
        
    def on_logging_stopped(self):
        """Handle logging stopped event."""
        self.logging_active = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Logging stopped")
        
        # Add stop message to log
        self.add_log_message("=== BLE Logging Stopped ===")
        
    def handle_ble_data(self, data: Dict[str, Any]):
        """Handle received BLE data and update GUI."""
        try:
            self.data_count += 1
            self.counter_label.setText(f"Data packets received: {self.data_count}")
            
            # Format data for display
            formatted_data = self.format_ble_data(data)
            self.add_log_message(formatted_data)
            
        except Exception as e:
            self.logger.error(f"Error handling BLE data: {e}")
            
    def format_ble_data(self, data: Dict[str, Any]) -> str:
        """Format BLE data for display in the log."""
        gui_config = self.config.get('gui', {})
        parts = []
        
        # Timestamp
        if gui_config.get('show_timestamp', True):
            timestamp = data.get('timestamp', datetime.now())
            parts.append(f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}]")
            
        # Data packet counter
        data_count = data.get('data_count', 0)
        parts.append(f"#{data_count:04d}")
            
        # Address/ID
        if gui_config.get('show_address', True):
            address = data.get('address', 'N/A')
            parts.append(f"ID: {address}")
            
        # Name
        if gui_config.get('show_name', True):
            name = data.get('name', 'N/A')
            parts.append(f"Name: {name}")
            
        # RSSI
        if gui_config.get('show_rssi', True):
            rssi = data.get('rssi', 'N/A')
            parts.append(f"RSSI: {rssi} dBm")
            
        # Distance
        if gui_config.get('show_distance', True):
            distance = data.get('distance')
            if distance is not None:
                parts.append(f"Dist: {distance:.2f}m")
            else:
                parts.append("Dist: N/A")
                
        # Battery
        if gui_config.get('show_battery', True):
            battery = data.get('batteryLevel')
            if battery is not None:
                parts.append(f"Batt: {battery}")
            else:
                parts.append("Batt: N/A")
                
        # Accelerometer
        acc_sensor = data.get('acc_sensor')
        if acc_sensor and gui_config.get('show_accelerometer', True):
            parts.append(
                f"Acc: X:{acc_sensor.x:.3f}g Y:{acc_sensor.y:.3f}g Z:{acc_sensor.z:.3f}g "
                f"Norm:{acc_sensor.norm:.3f}g"
            )
            
            # Movement detection
            if gui_config.get('show_movement', True):
                movement_status = "MOVING" if acc_sensor.is_moving else "STATIC"
                parts.append(f"Motion: {movement_status}")
                
        return " | ".join(parts)
        
    def add_log_message(self, message: str):
        """Add a message to the log display."""
        gui_config = self.config.get('gui', {})
        
        # Add message
        self.log_display.append(message)
        
        # Limit number of lines
        max_lines = gui_config.get('log_max_lines', 1000)
        document = self.log_display.document()
        while document.blockCount() > max_lines:
            cursor = QTextCursor(document.begin())
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # Remove the newline
            
        # Auto-scroll to bottom
        if gui_config.get('auto_scroll', True):
            cursor = self.log_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.log_display.setTextCursor(cursor)
            
    def handle_ble_error(self, error_message: str):
        """Handle BLE errors."""
        self.logger.error(f"BLE Error: {error_message}")
        self.add_log_message(f"ERROR: {error_message}")
        self.show_error_popup("BLE Error", error_message)
        
    def show_error_popup(self, title: str, message: str):
        """Show an error popup dialog."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()
        
    def closeEvent(self, event):
        """Clean up when the window is closed."""
        if self.logging_active:
            self.stop_logging()
        event.accept()


def main():
    """Main entry point for the BLE Manual Test GUI."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("BLE Manual Test GUI")
    app.setApplicationVersion("1.0")
    
    window = BleManualTestGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()