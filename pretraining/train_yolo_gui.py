"""PySide6-based GUI for training YOLO models with custom datasets."""

import threading
import sys
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QFileDialog, QMessageBox,
    QSplitter, QScrollArea, QTextEdit, QSpinBox, QDoubleSpinBox, 
    QComboBox, QGroupBox, QFormLayout, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QProcess
from PySide6.QtGui import QFont, QPixmap, QImage, QTextCursor


@dataclass
class TrainingConfig:
    """Configuration parameters for YOLO training."""
    data_dir: str = ""
    model: str = "yolov8x.pt"
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    device: str = ""
    val_ratio: float = 0.2
    onnx_out: str = "yolov8_finetuned.onnx"


class TrainingSignals(QObject):
    """Signals for thread-safe training updates."""
    output_ready = Signal(str)
    training_finished = Signal(int)  # exit code


class TrainYoloGUI(QMainWindow):
    """PySide6 GUI for training YOLO models."""

    def __init__(self):
        """Create widgets and bind actions."""
        super().__init__()
        self.setWindowTitle("YOLO Training GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Configuration
        self.config = TrainingConfig()
        self.config_path = Path(__file__).parent / "sample_train_config.yaml"
        self.load_config()
        
        # Process management
        self.training_process = None
        self.is_training = False
        
        # Set up the UI
        self._setup_ui()
        
        # Set up signals for thread-safe updates
        self.training_signals = TrainingSignals()
        self.training_signals.output_ready.connect(self._append_log)
        self.training_signals.training_finished.connect(self._on_training_finished)

    def _setup_ui(self) -> None:
        """Set up the main user interface."""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create main splitter for layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Dataset selection group
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QFormLayout(dataset_group)
        
        # Images directory selection
        images_layout = QHBoxLayout()
        self.images_dir_edit = QLineEdit(self.config.data_dir)
        self.images_dir_edit.setPlaceholderText("Select directory containing images and labels")
        self.select_images_btn = QPushButton("Browse...")
        self.select_images_btn.clicked.connect(self.select_images_dir)
        images_layout.addWidget(self.images_dir_edit)
        images_layout.addWidget(self.select_images_btn)
        dataset_layout.addRow("Data Directory:", images_layout)
        
        # ONNX output path
        onnx_layout = QHBoxLayout()
        self.onnx_out_edit = QLineEdit(self.config.onnx_out)
        self.onnx_out_edit.setPlaceholderText("Output ONNX model path")
        self.select_onnx_btn = QPushButton("Browse...")
        self.select_onnx_btn.clicked.connect(self.select_onnx_output)
        onnx_layout.addWidget(self.onnx_out_edit)
        onnx_layout.addWidget(self.select_onnx_btn)
        dataset_layout.addRow("ONNX Output:", onnx_layout)
        
        left_layout.addWidget(dataset_group)
        
        # Training parameters group
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo.setCurrentText(self.config.model)
        params_layout.addRow("Base Model:", self.model_combo)
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(1000)
        self.epochs_spin.setValue(self.config.epochs)
        params_layout.addRow("Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(256)
        self.batch_spin.setValue(self.config.batch)
        params_layout.addRow("Batch Size:", self.batch_spin)
        
        # Image size
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "800", "1024"])
        self.imgsz_combo.setCurrentText(str(self.config.imgsz))
        params_layout.addRow("Image Size:", self.imgsz_combo)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0", "1", "2", "3"])
        device_text = self.config.device if self.config.device else "auto"
        self.device_combo.setCurrentText(device_text)
        params_layout.addRow("Device:", self.device_combo)
        
        # Validation ratio
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setMinimum(0.1)
        self.val_ratio_spin.setMaximum(0.9)
        self.val_ratio_spin.setSingleStep(0.1)
        self.val_ratio_spin.setDecimals(1)
        self.val_ratio_spin.setValue(self.config.val_ratio)
        params_layout.addRow("Validation Ratio:", self.val_ratio_spin)
        
        left_layout.addWidget(params_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_config_btn)
        
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_config_dialog)
        button_layout.addWidget(self.load_config_btn)
        
        button_layout.addStretch()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("QPushButton { background-color: green; color: white; font-weight: bold; }")
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        button_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: lightgray;")
        left_layout.addWidget(self.status_label)
        
        # Add left panel to splitter
        main_splitter.addWidget(left_panel)
        
        # Right panel - Log output
        log_panel = QFrame()
        log_layout = QVBoxLayout(log_panel)
        
        log_label = QLabel("Training Log Output")
        log_label.setAlignment(Qt.AlignCenter)
        log_label.setStyleSheet("font-weight: bold; padding: 5px;")
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: black; color: white;")
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)
        
        # Add log panel to splitter
        main_splitter.addWidget(log_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([500, 700])

    def select_images_dir(self) -> None:
        """Prompt the user to select the images/labels directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select directory containing images and labels",
            self.config.data_dir
        )
        if directory:
            self.config.data_dir = directory
            self.images_dir_edit.setText(directory)

    def select_onnx_output(self) -> None:
        """Prompt the user to select the ONNX output file path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select ONNX output path",
            self.config.onnx_out,
            "ONNX files (*.onnx);;All files (*.*)"
        )
        if file_path:
            self.config.onnx_out = file_path
            self.onnx_out_edit.setText(file_path)

    def _update_config_from_ui(self) -> None:
        """Update the configuration object from UI values."""
        self.config.data_dir = self.images_dir_edit.text()
        self.config.model = self.model_combo.currentText()
        self.config.epochs = self.epochs_spin.value()
        self.config.batch = self.batch_spin.value()
        self.config.imgsz = int(self.imgsz_combo.currentText())
        device_text = self.device_combo.currentText()
        self.config.device = "" if device_text == "auto" else device_text
        self.config.val_ratio = self.val_ratio_spin.value()
        self.config.onnx_out = self.onnx_out_edit.text()

    def _update_ui_from_config(self) -> None:
        """Update UI values from the configuration object."""
        self.images_dir_edit.setText(self.config.data_dir)
        self.model_combo.setCurrentText(self.config.model)
        self.epochs_spin.setValue(self.config.epochs)
        self.batch_spin.setValue(self.config.batch)
        self.imgsz_combo.setCurrentText(str(self.config.imgsz))
        device_text = self.config.device if self.config.device else "auto"
        self.device_combo.setCurrentText(device_text)
        self.val_ratio_spin.setValue(self.config.val_ratio)
        self.onnx_out_edit.setText(self.config.onnx_out)

    def save_config(self) -> None:
        """Save current configuration to YAML file."""
        self._update_config_from_ui()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save configuration",
            str(self.config_path),
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    yaml.safe_dump(asdict(self.config), f, default_flow_style=False)
                self.status_label.setText(f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error saving config", str(e))

    def load_config_dialog(self) -> None:
        """Load configuration from YAML file via dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load configuration",
            str(self.config_path),
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if file_path:
            self.load_config(file_path)

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = str(self.config_path)
        
        if not os.path.exists(config_path):
            # Create default config if it doesn't exist
            self.save_default_config(config_path)
            return
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Update config with loaded values
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self._update_ui_from_config()
            self.status_label.setText(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error loading config", str(e))

    def save_default_config(self, config_path: str) -> None:
        """Save the default configuration to a file."""
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(asdict(self.config), f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Could not save default config: {e}")

    def start_training(self) -> None:
        """Start the YOLO training process."""
        if self.is_training:
            return
        
        # Validate inputs
        self._update_config_from_ui()
        
        if not self.config.data_dir:
            QMessageBox.critical(self, "Error", "Please select a data directory")
            return
        
        if not os.path.exists(self.config.data_dir):
            QMessageBox.critical(self, "Error", "Data directory does not exist")
            return
        
        if not self.config.onnx_out:
            QMessageBox.critical(self, "Error", "Please specify an ONNX output path")
            return
        
        # Prepare command
        train_script = Path(__file__).parent / "train_yolo.py"
        if not train_script.exists():
            QMessageBox.critical(self, "Error", f"train_yolo.py not found at {train_script}")
            return
        
        cmd = [
            sys.executable, str(train_script),
            self.config.data_dir,
            "--model", self.config.model,
            "--epochs", str(self.config.epochs),
            "--batch", str(self.config.batch),
            "--imgsz", str(self.config.imgsz),
            "--val-ratio", str(self.config.val_ratio),
            "--onnx-out", self.config.onnx_out
        ]
        
        if self.config.device:
            cmd.extend(["--device", self.config.device])
        
        # Start training in background thread
        self.is_training = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training in progress...")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: orange;")
        
        self.log_text.append(f"Starting training with command:\n{' '.join(cmd)}\n")
        
        # Use QProcess for better integration
        self.training_process = QProcess(self)
        self.training_process.readyReadStandardOutput.connect(self._read_stdout)
        self.training_process.readyReadStandardError.connect(self._read_stderr)
        self.training_process.finished.connect(self._on_training_finished)
        self.training_process.start(cmd[0], cmd[1:])

    def _read_stdout(self) -> None:
        """Read and display stdout from training process."""
        if self.training_process:
            data = self.training_process.readAllStandardOutput().data().decode()
            if data:
                self.training_signals.output_ready.emit(data)

    def _read_stderr(self) -> None:
        """Read and display stderr from training process."""
        if self.training_process:
            data = self.training_process.readAllStandardError().data().decode()
            if data:
                self.training_signals.output_ready.emit(data)

    def _append_log(self, text: str) -> None:
        """Append text to the log output."""
        self.log_text.append(text.rstrip())
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def stop_training(self) -> None:
        """Stop the training process."""
        if self.training_process and self.training_process.state() != QProcess.NotRunning:
            self.training_process.kill()
            self.log_text.append("\n=== Training stopped by user ===\n")

    def _on_training_finished(self, exit_code: int) -> None:
        """Handle training process completion."""
        self.is_training = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if exit_code == 0:
            self.status_label.setText("Training completed successfully")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: lightgreen;")
            self.log_text.append("\n=== Training completed successfully ===\n")
        else:
            self.status_label.setText(f"Training failed (exit code: {exit_code})")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: lightcoral;")
            self.log_text.append(f"\n=== Training failed (exit code: {exit_code}) ===\n")
        
        self.training_process = None


def create_app():
    """Create and return a QApplication instance for testing."""
    return QApplication(sys.argv if sys.argv else [])


def main():
    """Entry point for running the training GUI."""
    app = create_app()
    
    window = TrainYoloGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()