"""PySide6-based GUI for YOLOv8 training."""

import sys
import os
import threading
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QFileDialog, QMessageBox,
    QSplitter, QScrollArea, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QFont, QTextCursor


class LogSignals(QObject):
    """Signals for thread-safe log updates."""
    log_updated = Signal(str)
    training_finished = Signal(bool)  # True for success, False for error


class TrainingWorker(QThread):
    """Worker thread to run training subprocess and capture output."""
    
    def __init__(self, train_script_path: str, args: list):
        super().__init__()
        self.train_script_path = train_script_path
        self.args = args
        self.signals = LogSignals()
        
    def run(self):
        """Execute training subprocess and emit log updates."""
        try:
            cmd = [sys.executable, self.train_script_path] + self.args
            self.signals.log_updated.emit(f"Executing: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.signals.log_updated.emit(line)
            
            process.wait()
            
            if process.returncode == 0:
                self.signals.log_updated.emit("\n✅ Training completed successfully!\n")
                self.signals.training_finished.emit(True)
            else:
                self.signals.log_updated.emit(f"\n❌ Training failed with exit code {process.returncode}\n")
                self.signals.training_finished.emit(False)
                
        except Exception as e:
            self.signals.log_updated.emit(f"\n❌ Error running training: {str(e)}\n")
            self.signals.training_finished.emit(False)


class TrainYoloGUI(QMainWindow):
    """PySide6 GUI for YOLOv8 training."""
    
    def __init__(self):
        """Initialize the training GUI."""
        super().__init__()
        self.setWindowTitle("YOLOv8 Training GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Configuration management
        self.config_path = Path(__file__).parent / "train_config.yaml"
        self.config = self.load_config()
        
        # Training worker
        self.training_worker = None
        self.is_training = False
        
        # Initialize UI
        self.init_ui()
        self.populate_from_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
        
        # Return default configuration
        return {
            'training': {
                'model': 'yolov8x.pt',
                'epochs': 50,
                'batch': 16,
                'imgsz': 640,
                'device': None,
                'val_ratio': 0.2,
                'onnx_out': 'yolov8_finetuned.onnx'
            },
            'paths': {
                'data_dir': '',
                'last_image_dir': '',
                'last_output_dir': ''
            }
        }
    
    def save_config(self):
        """Save current configuration to YAML file."""
        try:
            # Update config with current values
            self.config['training']['model'] = self.model_combo.currentText()
            self.config['training']['epochs'] = self.epochs_spin.value()
            self.config['training']['batch'] = self.batch_spin.value()
            self.config['training']['imgsz'] = self.imgsz_spin.value()
            self.config['training']['device'] = self.device_edit.text() or None
            self.config['training']['val_ratio'] = self.val_ratio_spin.value()
            self.config['training']['onnx_out'] = self.onnx_out_edit.text()
            self.config['paths']['data_dir'] = self.data_dir_edit.text()
            
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def init_ui(self):
        """Initialize the user interface."""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Data directory selection
        data_group = QGroupBox("Dataset Configuration")
        data_layout = QVBoxLayout(data_group)
        
        # Data directory
        data_dir_layout = QHBoxLayout()
        data_dir_layout.addWidget(QLabel("Images & Labels Dir:"))
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("Select directory containing images and .txt labels")
        data_dir_layout.addWidget(self.data_dir_edit)
        
        self.select_data_btn = QPushButton("Browse...")
        self.select_data_btn.clicked.connect(self.select_data_directory)
        data_dir_layout.addWidget(self.select_data_btn)
        data_layout.addLayout(data_dir_layout)
        
        left_layout.addWidget(data_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
        self.model_combo.setEditable(True)
        params_layout.addRow("Model:", self.model_combo)
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        params_layout.addRow("Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        params_layout.addRow("Batch Size:", self.batch_spin)
        
        # Image size
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        params_layout.addRow("Image Size:", self.imgsz_spin)
        
        # Device
        self.device_edit = QLineEdit()
        self.device_edit.setPlaceholderText("Leave empty for auto-detection")
        params_layout.addRow("Device:", self.device_edit)
        
        # Validation ratio
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 1.0)
        self.val_ratio_spin.setSingleStep(0.1)
        self.val_ratio_spin.setValue(0.2)
        self.val_ratio_spin.setDecimals(2)
        params_layout.addRow("Val Ratio:", self.val_ratio_spin)
        
        # ONNX output path
        self.onnx_out_edit = QLineEdit()
        self.onnx_out_edit.setText("yolov8_finetuned.onnx")
        params_layout.addRow("ONNX Output:", self.onnx_out_edit)
        
        left_layout.addWidget(params_group)
        
        # Control buttons
        control_layout = QVBoxLayout()
        
        self.save_config_btn = QPushButton("Save Configuration")
        self.save_config_btn.clicked.connect(self.save_config)
        control_layout.addWidget(self.save_config_btn)
        
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self.start_training)
        self.start_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.start_training_btn)
        
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.stop_training_btn)
        
        left_layout.addLayout(control_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; padding: 5px; }")
        left_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        
        # Add left panel to splitter
        main_splitter.addWidget(left_panel)
        
        # Right panel - Real-time log display
        log_panel = QFrame()
        log_layout = QVBoxLayout(log_panel)
        
        log_label = QLabel("Training Log")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #555;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_btn)
        
        # Add log panel to splitter
        main_splitter.addWidget(log_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 800])
        
    def populate_from_config(self):
        """Populate UI fields from loaded configuration."""
        training_config = self.config.get('training', {})
        paths_config = self.config.get('paths', {})
        
        # Set training parameters
        model = training_config.get('model', 'yolov8x.pt')
        if model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(model)
        else:
            self.model_combo.setCurrentText(model)
            
        self.epochs_spin.setValue(training_config.get('epochs', 50))
        self.batch_spin.setValue(training_config.get('batch', 16))
        self.imgsz_spin.setValue(training_config.get('imgsz', 640))
        self.device_edit.setText(training_config.get('device') or '')
        self.val_ratio_spin.setValue(training_config.get('val_ratio', 0.2))
        self.onnx_out_edit.setText(training_config.get('onnx_out', 'yolov8_finetuned.onnx'))
        
        # Set paths
        self.data_dir_edit.setText(paths_config.get('data_dir', ''))
    
    def select_data_directory(self):
        """Open dialog to select data directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Images and Labels",
            self.config.get('paths', {}).get('last_image_dir', '')
        )
        if directory:
            self.data_dir_edit.setText(directory)
            # Update config for future use
            if 'paths' not in self.config:
                self.config['paths'] = {}
            self.config['paths']['last_image_dir'] = directory
    
    def validate_inputs(self) -> bool:
        """Validate user inputs before starting training."""
        if not self.data_dir_edit.text().strip():
            QMessageBox.critical(self, "Invalid Input", "Please select a data directory.")
            return False
            
        data_dir = Path(self.data_dir_edit.text())
        if not data_dir.exists():
            QMessageBox.critical(self, "Invalid Input", "Selected data directory does not exist.")
            return False
            
        # Check for images and labels
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = [f for f in data_dir.iterdir() if f.suffix.lower() in image_extensions]
        labels = [f for f in data_dir.iterdir() if f.suffix.lower() == '.txt']
        
        if not images:
            QMessageBox.critical(self, "Invalid Dataset", 
                               f"No images found in {data_dir}.\nSupported formats: {', '.join(image_extensions)}")
            return False
            
        if not labels:
            QMessageBox.critical(self, "Invalid Dataset", 
                               f"No YOLO label files (.txt) found in {data_dir}.")
            return False
            
        return True
    
    def start_training(self):
        """Start the training process."""
        if not self.validate_inputs():
            return
            
        if self.is_training:
            QMessageBox.warning(self, "Training in Progress", "Training is already running.")
            return
        
        # Save current configuration
        self.save_config()
        
        # Prepare arguments for train_yolo.py
        args = [
            str(Path(self.data_dir_edit.text()).resolve()),
            '--model', self.model_combo.currentText(),
            '--epochs', str(self.epochs_spin.value()),
            '--batch', str(self.batch_spin.value()),
            '--imgsz', str(self.imgsz_spin.value()),
            '--val-ratio', str(self.val_ratio_spin.value()),
            '--onnx-out', self.onnx_out_edit.text()
        ]
        
        if self.device_edit.text().strip():
            args.extend(['--device', self.device_edit.text().strip()])
        
        # Get the train_yolo.py script path
        train_script = Path(__file__).parent / "train_yolo.py"
        
        # Create and start training worker
        self.training_worker = TrainingWorker(str(train_script), args)
        self.training_worker.signals.log_updated.connect(self.append_log)
        self.training_worker.signals.training_finished.connect(self.training_finished)
        
        # Update UI for training state
        self.is_training = True
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.status_label.setText("Training in progress...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Clear log and start training
        self.clear_log()
        self.append_log("🚀 Starting YOLOv8 training...\n")
        self.training_worker.start()
    
    def stop_training(self):
        """Stop the training process."""
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.terminate()
            self.training_worker.wait(3000)  # Wait up to 3 seconds
            if self.training_worker.isRunning():
                self.training_worker.kill()  # Force kill if still running
            
        self.training_finished(False)
        self.append_log("\n⏹️ Training stopped by user.\n")
    
    def training_finished(self, success: bool):
        """Handle training completion."""
        self.is_training = False
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("Training completed successfully!")
        else:
            self.status_label.setText("Training failed or was interrupted.")
    
    def append_log(self, text: str):
        """Append text to the log display."""
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()
    
    def clear_log(self):
        """Clear the log display."""
        self.log_text.clear()
    
    def closeEvent(self, event):
        """Handle application close event."""
        if self.is_training:
            reply = QMessageBox.question(
                self, 'Training in Progress',
                'Training is currently running. Do you want to stop it and exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.stop_training()
                event.accept()
            else:
                event.ignore()
        else:
            self.save_config()
            event.accept()


def create_app():
    """Create and return a QApplication instance for headless testing."""
    return QApplication(sys.argv if sys.argv else [])


def main():
    """Entry point for running the training GUI."""
    app = create_app()
    
    window = TrainYoloGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()