"""PySide6-based GUI for YOLO model training."""

import threading
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QFileDialog, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QScrollArea, QMenuBar
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QAction


class TrainingSignals(QObject):
    """Signals for thread-safe training updates."""
    output_ready = Signal(str)
    training_finished = Signal(bool, str)  # success, message


class TrainYoloGUI(QMainWindow):
    """PySide6 GUI for YOLO model training."""

    def __init__(self):
        """Create widgets and bind actions."""
        super().__init__()
        self.setWindowTitle("YOLO Model Training")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize variables
        self.data_dir = ""
        self.training_process = None
        self.is_training = False
        self.config_file = Path(__file__).parent / "train_config.yaml"
        
        # Create menu bar
        self._create_menu_bar()
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("YOLO Model Training")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Data directory selection
        data_section = QFrame()
        data_section.setFrameStyle(QFrame.Box)
        data_layout = QVBoxLayout(data_section)
        
        data_title = QLabel("Dataset Configuration")
        data_title.setFont(QFont("", 12, QFont.Bold))
        data_layout.addWidget(data_title)
        
        # Data directory selection
        data_dir_layout = QHBoxLayout()
        data_dir_layout.addWidget(QLabel("Images & Annotations Directory:"))
        self.select_data_btn = QPushButton("Select Directory")
        self.select_data_btn.clicked.connect(self.select_data_directory)
        data_dir_layout.addWidget(self.select_data_btn)
        data_layout.addLayout(data_dir_layout)
        
        self.data_dir_label = QLabel("No directory selected")
        self.data_dir_label.setWordWrap(True)
        self.data_dir_label.setStyleSheet("color: gray; font-style: italic;")
        data_layout.addWidget(self.data_dir_label)
        
        main_layout.addWidget(data_section)
        
        # Training parameters
        params_section = QFrame()
        params_section.setFrameStyle(QFrame.Box)
        params_layout = QVBoxLayout(params_section)
        
        params_title = QLabel("Training Parameters")
        params_title.setFont(QFont("", 12, QFont.Bold))
        params_layout.addWidget(params_title)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Base Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo.setCurrentText("yolov8x.pt")
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        params_layout.addLayout(model_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(1000)
        self.epochs_spin.setValue(50)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        params_layout.addLayout(epochs_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(128)
        self.batch_spin.setValue(16)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        params_layout.addLayout(batch_layout)
        
        # Image size
        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(QLabel("Image Size:"))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setMinimum(32)
        self.imgsz_spin.setMaximum(1024)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        imgsz_layout.addWidget(self.imgsz_spin)
        imgsz_layout.addStretch()
        params_layout.addLayout(imgsz_layout)
        
        # Validation ratio
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation Ratio:"))
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setMinimum(0.05)
        self.val_ratio_spin.setMaximum(0.5)
        self.val_ratio_spin.setValue(0.2)
        self.val_ratio_spin.setSingleStep(0.05)
        val_layout.addWidget(self.val_ratio_spin)
        val_layout.addStretch()
        params_layout.addLayout(val_layout)
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0", "1"])
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        params_layout.addLayout(device_layout)
        
        # Output ONNX path
        onnx_layout = QHBoxLayout()
        onnx_layout.addWidget(QLabel("Output ONNX Name:"))
        self.onnx_edit = QLineEdit("yolov8_finetuned.onnx")
        onnx_layout.addWidget(self.onnx_edit)
        params_layout.addLayout(onnx_layout)
        
        main_layout.addWidget(params_section)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # Status
        self.status_label = QLabel("Ready to start training")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Output log
        log_label = QLabel("Training Output:")
        log_label.setFont(QFont("", 10, QFont.Bold))
        main_layout.addWidget(log_label)
        
        self.output_text = QTextEdit()
        self.output_text.setMaximumHeight(200)
        self.output_text.setFont(QFont("Consolas", 9))
        main_layout.addWidget(self.output_text)
        
        # Set up signals for thread-safe updates
        self.training_signals = TrainingSignals()
        self.training_signals.output_ready.connect(self._update_output)
        self.training_signals.training_finished.connect(self._training_finished)
        
        # Load default configuration
        self._load_config_if_exists()

    def select_data_directory(self) -> None:
        """Prompt the user to select a directory containing images and annotations."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select directory containing images and YOLO .txt annotations"
        )
        if directory:
            self.data_dir = directory
            self.data_dir_label.setText(directory)
            self.data_dir_label.setStyleSheet("color: black;")

    def start_training(self) -> None:
        """Launch training in a background thread."""
        if not self.data_dir:
            QMessageBox.critical(
                self,
                "Missing Data Directory",
                "Please select a directory containing images and annotations"
            )
            return
            
        # Validate data directory contains images and labels
        data_path = Path(self.data_dir)
        image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
        label_files = list(data_path.glob("*.txt"))
        
        if not image_files:
            QMessageBox.critical(
                self,
                "No Images Found",
                "The selected directory doesn't contain any .jpg or .png image files"
            )
            return
            
        if not label_files:
            QMessageBox.critical(
                self,
                "No Labels Found", 
                "The selected directory doesn't contain any .txt annotation files"
            )
            return
        
        # Update UI state
        self.is_training = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training in progress...")
        self.output_text.clear()
        
        # Start training in background thread
        thread = threading.Thread(
            target=self._run_training_thread,
            daemon=True
        )
        thread.start()

    def _run_training_thread(self) -> None:
        """Execute training and emit signals for UI updates."""
        try:
            # Build command
            train_script = Path(__file__).parent / "train_yolo.py"
            cmd = [
                sys.executable, str(train_script),
                self.data_dir,
                "--model", self.model_combo.currentText(),
                "--epochs", str(self.epochs_spin.value()),
                "--batch", str(self.batch_spin.value()),
                "--imgsz", str(self.imgsz_spin.value()),
                "--val-ratio", str(self.val_ratio_spin.value()),
                "--onnx-out", self.onnx_edit.text()
            ]
            
            # Add device if not auto
            if self.device_combo.currentText() != "auto":
                cmd.extend(["--device", self.device_combo.currentText()])
            
            # Run training process
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            for line in self.training_process.stdout:
                if not self.is_training:  # Check if stopped
                    break
                self.training_signals.output_ready.emit(line.rstrip())
            
            # Wait for completion
            return_code = self.training_process.wait()
            
            if return_code == 0:
                self.training_signals.training_finished.emit(True, "Training completed successfully!")
            else:
                self.training_signals.training_finished.emit(False, f"Training failed with exit code {return_code}")
                
        except Exception as e:
            self.training_signals.training_finished.emit(False, f"Training error: {str(e)}")

    def stop_training(self) -> None:
        """Stop the current training process."""
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
        
        self.is_training = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Training stopped by user")

    def _update_output(self, line: str) -> None:
        """Update the output text area with new training output."""
        self.output_text.append(line)
        # Auto-scroll to bottom
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _training_finished(self, success: bool, message: str) -> None:
        """Handle training completion."""
        self.is_training = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(message)
        
        if success:
            QMessageBox.information(self, "Training Complete", message)
        else:
            QMessageBox.critical(self, "Training Failed", message)

    def _create_menu_bar(self) -> None:
        """Create the menu bar with config options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Load config action
        load_action = QAction("Load Config...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_config_dialog)
        file_menu.addAction(load_action)
        
        # Save config action
        save_action = QAction("Save Config...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_config_dialog)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Save as default config action
        save_default_action = QAction("Save as Default Config", self)
        save_default_action.triggered.connect(self._save_default_config)
        file_menu.addAction(save_default_action)

    def _get_current_config(self) -> Dict[str, Any]:
        """Get current GUI parameters as a config dictionary."""
        return {
            "data_dir": self.data_dir,
            "model": self.model_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "device": self.device_combo.currentText(),
            "val_ratio": self.val_ratio_spin.value(),
            "onnx_out": self.onnx_edit.text()
        }

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration values to GUI elements."""
        if "data_dir" in config and config["data_dir"]:
            self.data_dir = config["data_dir"]
            self.data_dir_label.setText(config["data_dir"])
            self.data_dir_label.setStyleSheet("color: black;")
        
        if "model" in config:
            self.model_combo.setCurrentText(config["model"])
        
        if "epochs" in config:
            self.epochs_spin.setValue(config["epochs"])
        
        if "batch" in config:
            self.batch_spin.setValue(config["batch"])
        
        if "imgsz" in config:
            self.imgsz_spin.setValue(config["imgsz"])
        
        if "device" in config:
            self.device_combo.setCurrentText(config["device"])
        
        if "val_ratio" in config:
            self.val_ratio_spin.setValue(config["val_ratio"])
        
        if "onnx_out" in config:
            self.onnx_edit.setText(config["onnx_out"])

    def _load_config_if_exists(self) -> None:
        """Load default configuration if it exists."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self._apply_config(config)
            except Exception as e:
                # Ignore errors loading default config
                pass

    def _load_config_dialog(self) -> None:
        """Show dialog to load configuration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Training Configuration",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self._apply_config(config)
                QMessageBox.information(
                    self, "Config Loaded", 
                    f"Configuration loaded from {Path(file_path).name}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Load Error", 
                    f"Failed to load configuration:\n{str(e)}"
                )

    def _save_config_dialog(self) -> None:
        """Show dialog to save current configuration to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Configuration",
            str(Path.cwd() / "train_config.yaml"),
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if file_path:
            try:
                config = self._get_current_config()
                with open(file_path, 'w') as f:
                    yaml.safe_dump(config, f, default_flow_style=False)
                QMessageBox.information(
                    self, "Config Saved", 
                    f"Configuration saved to {Path(file_path).name}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", 
                    f"Failed to save configuration:\n{str(e)}"
                )

    def _save_default_config(self) -> None:
        """Save current configuration as the default."""
        try:
            config = self._get_current_config()
            with open(self.config_file, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            QMessageBox.information(
                self, "Default Config Saved", 
                "Current configuration saved as default"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error", 
                f"Failed to save default configuration:\n{str(e)}"
            )


def create_app():
    """Create and return a QApplication instance for headless testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if sys.argv else [])
    return app


def main():
    """Entry point for running the training GUI."""
    app = create_app()
    
    window = TrainYoloGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()