"""Tests for the YOLO training GUI non-GUI logic."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the GUI module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "pretraining"))

from train_yolo_gui import TrainingConfig, TrainYoloGUI


class TestTrainingConfig:
    """Test the TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = TrainingConfig()
        assert config.data_dir == ""
        assert config.model == "yolov8x.pt"
        assert config.epochs == 50
        assert config.batch == 16
        assert config.imgsz == 640
        assert config.device == ""
        assert config.val_ratio == 0.2
        assert config.onnx_out == "yolov8_finetuned.onnx"

    def test_custom_values(self):
        """Test that custom configuration values can be set."""
        config = TrainingConfig(
            data_dir="/test/data",
            model="yolov8n.pt",
            epochs=100,
            batch=8,
            imgsz=1024,
            device="0",
            val_ratio=0.1,
            onnx_out="custom.onnx"
        )
        assert config.data_dir == "/test/data"
        assert config.model == "yolov8n.pt"
        assert config.epochs == 100
        assert config.batch == 8
        assert config.imgsz == 1024
        assert config.device == "0"
        assert config.val_ratio == 0.1
        assert config.onnx_out == "custom.onnx"


class TestTrainYoloGUILogic:
    """Test the non-GUI logic of TrainYoloGUI."""

    @patch('train_yolo_gui.QApplication')
    @patch('train_yolo_gui.QMainWindow.__init__')
    def setup_method(self, method, mock_main_window, mock_qapp):
        """Set up test fixtures."""
        # Mock the Qt components to avoid GUI initialization
        mock_main_window.return_value = None
        with patch.object(TrainYoloGUI, '_setup_ui'):
            self.gui = TrainYoloGUI()
            # training_signals is created in __init__, so we mock it after instantiation
            self.gui.training_signals = Mock()

    def test_update_config_from_ui(self):
        """Test updating configuration from UI elements."""
        # Mock UI elements
        self.gui.images_dir_edit = Mock(text=lambda: "/test/images")
        self.gui.model_combo = Mock(currentText=lambda: "yolov8n.pt")
        self.gui.epochs_spin = Mock(value=lambda: 100)
        self.gui.batch_spin = Mock(value=lambda: 8)
        self.gui.imgsz_combo = Mock(currentText=lambda: "1024")
        self.gui.device_combo = Mock(currentText=lambda: "0")
        self.gui.val_ratio_spin = Mock(value=lambda: 0.1)
        self.gui.onnx_out_edit = Mock(text=lambda: "custom.onnx")

        self.gui._update_config_from_ui()

        assert self.gui.config.data_dir == "/test/images"
        assert self.gui.config.model == "yolov8n.pt"
        assert self.gui.config.epochs == 100
        assert self.gui.config.batch == 8
        assert self.gui.config.imgsz == 1024
        assert self.gui.config.device == "0"
        assert self.gui.config.val_ratio == 0.1
        assert self.gui.config.onnx_out == "custom.onnx"

    def test_update_config_from_ui_auto_device(self):
        """Test that 'auto' device selection becomes empty string."""
        self.gui.images_dir_edit = Mock(text=lambda: "/test")
        self.gui.model_combo = Mock(currentText=lambda: "yolov8x.pt")
        self.gui.epochs_spin = Mock(value=lambda: 50)
        self.gui.batch_spin = Mock(value=lambda: 16)
        self.gui.imgsz_combo = Mock(currentText=lambda: "640")
        self.gui.device_combo = Mock(currentText=lambda: "auto")
        self.gui.val_ratio_spin = Mock(value=lambda: 0.2)
        self.gui.onnx_out_edit = Mock(text=lambda: "test.onnx")

        self.gui._update_config_from_ui()

        assert self.gui.config.device == ""

    def test_save_config(self):
        """Test saving configuration to YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Mock UI elements
            self.gui.images_dir_edit = Mock(text=lambda: "/test/data")
            self.gui.model_combo = Mock(currentText=lambda: "yolov8s.pt")
            self.gui.epochs_spin = Mock(value=lambda: 25)
            self.gui.batch_spin = Mock(value=lambda: 32)
            self.gui.imgsz_combo = Mock(currentText=lambda: "512")
            self.gui.device_combo = Mock(currentText=lambda: "cpu")
            self.gui.val_ratio_spin = Mock(value=lambda: 0.3)
            self.gui.onnx_out_edit = Mock(text=lambda: "output.onnx")
            self.gui.status_label = Mock()

            # Mock QFileDialog to return our temp file
            with patch('train_yolo_gui.QFileDialog.getSaveFileName', return_value=(temp_path, "")):
                self.gui.save_config()

            # Verify file was written correctly
            with open(temp_path, 'r') as f:
                saved_config = yaml.safe_load(f)

            assert saved_config['data_dir'] == "/test/data"
            assert saved_config['model'] == "yolov8s.pt"
            assert saved_config['epochs'] == 25
            assert saved_config['batch'] == 32
            assert saved_config['imgsz'] == 512
            assert saved_config['device'] == "cpu"
            assert saved_config['val_ratio'] == 0.3
            assert saved_config['onnx_out'] == "output.onnx"

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def test_load_config(self):
        """Test loading configuration from YAML file."""
        # Create temporary config file
        config_data = {
            'data_dir': '/loaded/data',
            'model': 'yolov8m.pt',
            'epochs': 75,
            'batch': 4,
            'imgsz': 800,
            'device': '1',
            'val_ratio': 0.15,
            'onnx_out': 'loaded.onnx'
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            # Mock UI update method
            self.gui._update_ui_from_config = Mock()
            self.gui.status_label = Mock()

            self.gui.load_config(temp_path)

            # Verify config was loaded
            assert self.gui.config.data_dir == "/loaded/data"
            assert self.gui.config.model == "yolov8m.pt"
            assert self.gui.config.epochs == 75
            assert self.gui.config.batch == 4
            assert self.gui.config.imgsz == 800
            assert self.gui.config.device == "1"
            assert self.gui.config.val_ratio == 0.15
            assert self.gui.config.onnx_out == "loaded.onnx"

            # Verify UI update was called
            self.gui._update_ui_from_config.assert_called_once()

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nonexistent_config(self):
        """Test loading configuration when file doesn't exist creates default."""
        nonexistent_path = "/nonexistent/config.yaml"
        
        self.gui.save_default_config = Mock()
        self.gui.load_config(nonexistent_path)
        
        self.gui.save_default_config.assert_called_once_with(nonexistent_path)

    @patch('train_yolo_gui.os.path.exists')
    @patch('train_yolo_gui.sys.executable', '/usr/bin/python')
    def test_training_command_generation(self, mock_exists):
        """Test that the training command is generated correctly."""
        mock_exists.return_value = True
        
        # Set up config
        self.gui.config.data_dir = "/test/data"
        self.gui.config.model = "yolov8n.pt"
        self.gui.config.epochs = 10
        self.gui.config.batch = 2
        self.gui.config.imgsz = 416
        self.gui.config.device = "cpu"
        self.gui.config.val_ratio = 0.1
        self.gui.config.onnx_out = "test.onnx"
        
        # Mock UI elements for validation
        self.gui._update_config_from_ui = Mock()
        self.gui.start_btn = Mock()
        self.gui.stop_btn = Mock()
        self.gui.status_label = Mock()
        self.gui.log_text = Mock()
        
        # Mock QProcess
        with patch('train_yolo_gui.QProcess') as mock_qprocess:
            mock_process = Mock()
            mock_qprocess.return_value = mock_process
            
            # Mock the train_yolo.py file existence
            train_script_path = Path(__file__).parent.parent / "pretraining" / "train_yolo.py"
            with patch.object(Path, 'exists', return_value=True):
                self.gui.start_training()
            
            # Verify process was started with correct arguments
            mock_process.start.assert_called_once()
            args = mock_process.start.call_args[0]
            
            expected_args = [
                str(train_script_path),
                "/test/data",
                "--model", "yolov8n.pt",
                "--epochs", "10",
                "--batch", "2",
                "--imgsz", "416",
                "--val-ratio", "0.1",
                "--onnx-out", "test.onnx",
                "--device", "cpu"
            ]
            
            assert args[0] == '/usr/bin/python'
            assert args[1] == expected_args

    def test_append_log(self):
        """Test log text appending functionality."""
        # Mock QTextEdit and QTextCursor
        self.gui.log_text = Mock()
        mock_cursor = Mock()
        self.gui.log_text.textCursor.return_value = mock_cursor
        
        test_text = "Test log message\nwith newlines\n"
        self.gui._append_log(test_text)
        
        # Verify text was appended without trailing newline
        self.gui.log_text.append.assert_called_once_with("Test log message\nwith newlines")
        
        # Verify cursor was moved to end
        mock_cursor.movePosition.assert_called_once()
        self.gui.log_text.setTextCursor.assert_called_once_with(mock_cursor)


# Headless test runner
def test_headless_functionality():
    """Test that all non-GUI logic works correctly without Qt initialization."""
    # Test TrainingConfig
    config = TrainingConfig()
    assert config.epochs == 50
    
    # Test config with custom values
    custom_config = TrainingConfig(epochs=100, batch=8)
    assert custom_config.epochs == 100
    assert custom_config.batch == 8
    
    # Test YAML serialization
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump({'epochs': 123, 'batch': 456}, f)
        temp_path = f.name
    
    try:
        with open(temp_path, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded['epochs'] == 123
        assert loaded['batch'] == 456
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])