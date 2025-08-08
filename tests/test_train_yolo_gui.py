"""Tests for the YOLOv8 training GUI (headless)."""

import unittest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the GUI module
import sys
sys.path.append(str(Path(__file__).parent.parent / "pretraining"))

from train_yolo_gui import TrainYoloGUI, create_app


class TestTrainYoloGUIHeadless(unittest.TestCase):
    """Test the non-GUI logic of the training GUI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app()
        
        # Create temporary config file
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
        
        # Sample configuration
        self.sample_config = {
            'training': {
                'model': 'yolov8n.pt',
                'epochs': 10,
                'batch': 8,
                'imgsz': 320,
                'device': 'cpu',
                'val_ratio': 0.15,
                'onnx_out': 'test_model.onnx'
            },
            'paths': {
                'data_dir': '/test/data',
                'last_image_dir': '/test/images',
                'last_output_dir': '/test/output'
            }
        }
        
        # Write sample config
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.sample_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration loading functionality."""
        # Patch the config path
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Check that configuration was loaded correctly
            self.assertEqual(gui.config['training']['model'], 'yolov8n.pt')
            self.assertEqual(gui.config['training']['epochs'], 10)
            self.assertEqual(gui.config['training']['batch'], 8)
            self.assertEqual(gui.config['paths']['data_dir'], '/test/data')
    
    def test_config_loading_with_missing_file(self):
        """Test configuration loading when file doesn't exist."""
        missing_path = self.temp_dir / "missing_config.yaml"
        
        with patch.object(TrainYoloGUI, 'config_path', missing_path):
            gui = TrainYoloGUI()
            
            # Check that default configuration is used
            self.assertEqual(gui.config['training']['model'], 'yolov8x.pt')
            self.assertEqual(gui.config['training']['epochs'], 50)
            self.assertEqual(gui.config['training']['batch'], 16)
    
    def test_config_saving(self):
        """Test configuration saving functionality."""
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Mock UI elements with test values
            gui.model_combo = MagicMock()
            gui.model_combo.currentText.return_value = 'yolov8s.pt'
            
            gui.epochs_spin = MagicMock()
            gui.epochs_spin.value.return_value = 25
            
            gui.batch_spin = MagicMock()
            gui.batch_spin.value.return_value = 32
            
            gui.imgsz_spin = MagicMock()
            gui.imgsz_spin.value.return_value = 416
            
            gui.device_edit = MagicMock()
            gui.device_edit.text.return_value = 'cuda'
            
            gui.val_ratio_spin = MagicMock()
            gui.val_ratio_spin.value.return_value = 0.3
            
            gui.onnx_out_edit = MagicMock()
            gui.onnx_out_edit.text.return_value = 'custom_model.onnx'
            
            gui.data_dir_edit = MagicMock()
            gui.data_dir_edit.text.return_value = '/custom/data'
            
            # Save configuration
            gui.save_config()
            
            # Load and verify saved configuration
            with open(self.config_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            self.assertEqual(saved_config['training']['model'], 'yolov8s.pt')
            self.assertEqual(saved_config['training']['epochs'], 25)
            self.assertEqual(saved_config['training']['batch'], 32)
            self.assertEqual(saved_config['training']['imgsz'], 416)
            self.assertEqual(saved_config['training']['device'], 'cuda')
            self.assertEqual(saved_config['training']['val_ratio'], 0.3)
            self.assertEqual(saved_config['training']['onnx_out'], 'custom_model.onnx')
            self.assertEqual(saved_config['paths']['data_dir'], '/custom/data')
    
    def test_input_validation_missing_directory(self):
        """Test input validation with missing data directory."""
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Mock empty data directory
            gui.data_dir_edit = MagicMock()
            gui.data_dir_edit.text.return_value = ''
            
            # Validation should fail
            self.assertFalse(gui.validate_inputs())
    
    def test_input_validation_nonexistent_directory(self):
        """Test input validation with non-existent data directory."""
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Mock non-existent data directory
            gui.data_dir_edit = MagicMock()
            gui.data_dir_edit.text.return_value = '/nonexistent/directory'
            
            # Validation should fail
            self.assertFalse(gui.validate_inputs())
    
    def test_input_validation_valid_directory_no_images(self):
        """Test input validation with directory that has no images."""
        # Create test directory without images
        test_data_dir = self.temp_dir / "no_images"
        test_data_dir.mkdir()
        
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Mock data directory path
            gui.data_dir_edit = MagicMock()
            gui.data_dir_edit.text.return_value = str(test_data_dir)
            
            # Validation should fail (no images)
            self.assertFalse(gui.validate_inputs())
    
    def test_input_validation_valid_directory_no_labels(self):
        """Test input validation with directory that has images but no labels."""
        # Create test directory with images but no labels
        test_data_dir = self.temp_dir / "no_labels"
        test_data_dir.mkdir()
        
        # Create a test image
        (test_data_dir / "test.jpg").touch()
        
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Mock data directory path
            gui.data_dir_edit = MagicMock()
            gui.data_dir_edit.text.return_value = str(test_data_dir)
            
            # Validation should fail (no labels)
            self.assertFalse(gui.validate_inputs())
    
    def test_input_validation_valid_directory(self):
        """Test input validation with valid directory containing images and labels."""
        # Create test directory with images and labels
        test_data_dir = self.temp_dir / "valid_data"
        test_data_dir.mkdir()
        
        # Create test files
        (test_data_dir / "image1.jpg").touch()
        (test_data_dir / "image1.txt").touch()
        (test_data_dir / "image2.png").touch()
        (test_data_dir / "image2.txt").touch()
        
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Mock data directory path
            gui.data_dir_edit = MagicMock()
            gui.data_dir_edit.text.return_value = str(test_data_dir)
            
            # Validation should pass
            self.assertTrue(gui.validate_inputs())
    
    def test_populate_from_config(self):
        """Test UI population from configuration."""
        with patch.object(TrainYoloGUI, 'config_path', self.config_path):
            gui = TrainYoloGUI()
            
            # Mock UI elements
            gui.model_combo = MagicMock()
            gui.model_combo.itemText = MagicMock(side_effect=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
            gui.model_combo.count = MagicMock(return_value=5)
            gui.model_combo.setCurrentText = MagicMock()
            
            gui.epochs_spin = MagicMock()
            gui.batch_spin = MagicMock()
            gui.imgsz_spin = MagicMock()
            gui.device_edit = MagicMock()
            gui.val_ratio_spin = MagicMock()
            gui.onnx_out_edit = MagicMock()
            gui.data_dir_edit = MagicMock()
            
            # Call populate method
            gui.populate_from_config()
            
            # Verify UI elements were set with config values
            gui.model_combo.setCurrentText.assert_called_with('yolov8n.pt')
            gui.epochs_spin.setValue.assert_called_with(10)
            gui.batch_spin.setValue.assert_called_with(8)
            gui.imgsz_spin.setValue.assert_called_with(320)
            gui.device_edit.setText.assert_called_with('cpu')
            gui.val_ratio_spin.setValue.assert_called_with(0.15)
            gui.onnx_out_edit.setText.assert_called_with('test_model.onnx')
            gui.data_dir_edit.setText.assert_called_with('/test/data')


if __name__ == '__main__':
    unittest.main()