"""Tests for the YOLO training GUI."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent / "pretraining"))
from train_yolo_gui import TrainYoloGUI, create_app


class TestTrainYoloGUI:
    """Test cases for the YOLO training GUI."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        """Set up QApplication for GUI testing."""
        self.app = create_app()
        yield
        # Clean up after each test
        if self.app:
            self.app.processEvents()

    @pytest.fixture
    def gui(self):
        """Create a TrainYoloGUI instance for testing."""
        return TrainYoloGUI()

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with sample images and labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample image files
            for i in range(3):
                (temp_path / f"image_{i}.jpg").write_text("fake image data")
                
            # Create sample label files
            for i in range(3):
                (temp_path / f"image_{i}.txt").write_text("0 0.5 0.5 0.1 0.1")
                
            yield temp_path

    def test_gui_creation(self, gui):
        """Test that the GUI can be created without errors."""
        assert gui.windowTitle() == "YOLO Model Training"
        assert gui.data_dir == ""
        assert not gui.is_training

    def test_initial_ui_state(self, gui):
        """Test the initial state of UI components."""
        assert gui.start_btn.isEnabled()
        assert not gui.stop_btn.isEnabled()
        assert gui.status_label.text() == "Ready to start training"
        assert gui.model_combo.currentText() == "yolov8x.pt"
        assert gui.epochs_spin.value() == 50
        assert gui.batch_spin.value() == 16

    def test_select_data_directory(self, gui, temp_data_dir):
        """Test data directory selection functionality."""
        # Simulate selecting a directory
        gui.data_dir = str(temp_data_dir)
        gui.data_dir_label.setText(str(temp_data_dir))
        
        assert gui.data_dir == str(temp_data_dir)
        assert gui.data_dir_label.text() == str(temp_data_dir)

    def test_validation_no_data_directory(self, gui, qtbot):
        """Test validation when no data directory is selected."""
        with patch('PySide6.QtWidgets.QMessageBox.critical') as mock_critical:
            gui.start_training()
            mock_critical.assert_called_once()
            
        # Button should remain enabled since training didn't start
        assert gui.start_btn.isEnabled()
        assert not gui.stop_btn.isEnabled()

    def test_validation_no_images(self, gui, qtbot):
        """Test validation when directory has no images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Directory with no files
            gui.data_dir = temp_dir
            
            with patch('PySide6.QtWidgets.QMessageBox.critical') as mock_critical:
                gui.start_training()
                mock_critical.assert_called_once()
                
            assert gui.start_btn.isEnabled()

    def test_validation_no_labels(self, gui, qtbot):
        """Test validation when directory has images but no labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create only image files, no labels
            (temp_path / "image.jpg").write_text("fake image data")
            
            gui.data_dir = temp_dir
            
            with patch('PySide6.QtWidgets.QMessageBox.critical') as mock_critical:
                gui.start_training()
                mock_critical.assert_called_once()
                
            assert gui.start_btn.isEnabled()

    def test_parameter_configuration(self, gui):
        """Test that training parameters can be configured."""
        # Test model selection
        gui.model_combo.setCurrentText("yolov8n.pt")
        assert gui.model_combo.currentText() == "yolov8n.pt"
        
        # Test epochs
        gui.epochs_spin.setValue(100)
        assert gui.epochs_spin.value() == 100
        
        # Test batch size
        gui.batch_spin.setValue(8)
        assert gui.batch_spin.value() == 8
        
        # Test image size
        gui.imgsz_spin.setValue(512)
        assert gui.imgsz_spin.value() == 512
        
        # Test validation ratio
        gui.val_ratio_spin.setValue(0.3)
        assert gui.val_ratio_spin.value() == 0.3
        
        # Test device
        gui.device_combo.setCurrentText("cpu")
        assert gui.device_combo.currentText() == "cpu"
        
        # Test output name
        gui.onnx_edit.setText("custom_model.onnx")
        assert gui.onnx_edit.text() == "custom_model.onnx"

    @patch('pretraining.train_yolo_gui.subprocess.Popen')
    @patch('threading.Thread')
    def test_start_training_valid_data(self, mock_thread, mock_popen, gui, temp_data_dir):
        """Test starting training with valid data directory."""
        gui.data_dir = str(temp_data_dir)
        
        # Mock successful process
        mock_process = MagicMock()
        mock_process.stdout = ["Training line 1", "Training line 2"]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        gui.start_training()
        
        # Verify UI state changes
        assert not gui.start_btn.isEnabled()
        assert gui.stop_btn.isEnabled()
        assert gui.is_training
        assert "Training in progress" in gui.status_label.text()
        
        # Verify thread was started
        mock_thread.assert_called_once()

    def test_stop_training(self, gui):
        """Test stopping training functionality."""
        # Simulate training state
        gui.is_training = True
        gui.start_btn.setEnabled(False)
        gui.stop_btn.setEnabled(True)
        
        # Mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        gui.training_process = mock_process
        
        gui.stop_training()
        
        # Verify process termination
        mock_process.terminate.assert_called_once()
        
        # Verify UI state reset
        assert not gui.is_training
        assert gui.start_btn.isEnabled()
        assert not gui.stop_btn.isEnabled()
        assert "stopped" in gui.status_label.text().lower()

    def test_output_update(self, gui):
        """Test training output update functionality."""
        initial_text = gui.output_text.toPlainText()
        
        gui._update_output("Test training output line")
        
        updated_text = gui.output_text.toPlainText()
        assert "Test training output line" in updated_text
        assert len(updated_text) > len(initial_text)

    def test_training_finished_success(self, gui):
        """Test successful training completion handling."""
        gui.is_training = True
        gui.start_btn.setEnabled(False)
        gui.stop_btn.setEnabled(True)
        
        with patch('PySide6.QtWidgets.QMessageBox.information') as mock_info:
            gui._training_finished(True, "Training completed successfully!")
            mock_info.assert_called_once()
        
        # Verify UI state reset
        assert not gui.is_training
        assert gui.start_btn.isEnabled()
        assert not gui.stop_btn.isEnabled()
        assert "successfully" in gui.status_label.text()

    def test_training_finished_failure(self, gui):
        """Test failed training completion handling."""
        gui.is_training = True
        gui.start_btn.setEnabled(False)
        gui.stop_btn.setEnabled(True)
        
        with patch('PySide6.QtWidgets.QMessageBox.critical') as mock_critical:
            gui._training_finished(False, "Training failed!")
            mock_critical.assert_called_once()
        
        # Verify UI state reset
        assert not gui.is_training
        assert gui.start_btn.isEnabled()
        assert not gui.stop_btn.isEnabled()
        assert "failed" in gui.status_label.text()


def test_create_app():
    """Test create_app function."""
    app = create_app()
    assert app is not None
    assert isinstance(app, QApplication)