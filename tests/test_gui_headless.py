"""Headless tests for PyQt-based GUI components."""

import os
import sys
import pytest
from unittest.mock import Mock, patch

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code', 'Application'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code', 'Modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pretraining', 'annotation'))

# Handle PySide6 imports gracefully - if system dependencies are missing,
# pytest will skip all tests in this module during collection
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer  
    from PySide6.QtGui import QPixmap
    PYSIDE6_AVAILABLE = True
except ImportError as e:
    # Create placeholder classes to prevent NameError during test collection
    QApplication = None
    QTimer = None
    QPixmap = None
    PYSIDE6_AVAILABLE = False
    PYSIDE6_IMPORT_ERROR = str(e)

# Skip all tests if PySide6 dependencies are not available
pytestmark = pytest.mark.skipif(
    not PYSIDE6_AVAILABLE, 
    reason=f"PySide6 dependencies not available: {PYSIDE6_IMPORT_ERROR if not PYSIDE6_AVAILABLE else 'Unknown error'}"
)


def test_create_app_headless():
    """Test that QApplication can be created in headless mode."""
    # Set up headless environment
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    try:
        from GUI_PyQt import create_app
        app = create_app()
        assert app is not None
        assert isinstance(app, QApplication)
        app.quit()
    except ImportError:
        pytest.skip("GUI_PyQt not available - likely missing dependencies")


def test_bearvision_gui_creation_headless():
    """Test that BearVisionGUI can be instantiated in headless mode."""
    # Set up headless environment
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    try:
        from GUI_PyQt import BearVisionGUI, create_app
        
        # Mock the Application class and dependencies
        with patch('GUI_PyQt.ConfigurationHandler') as mock_config:
            mock_config.get_configuration.return_value = {
                "GUI": {"video_path": "/test/video", "user_path": "/test/user"},
                "MOTION_DETECTION": {"search_box_dimensions": "100,100,200,200"}
            }
            
            # Create app and GUI
            app = create_app()
            mock_app_ref = Mock()
            
            gui = BearVisionGUI(mock_app_ref)
            
            # Basic checks
            assert gui.windowTitle() == "BearVision - WakeVision"
            assert gui.video_folder_entry.text() == "/test/video"
            assert gui.user_folder_entry.text() == "/test/user"
            
            # Check that preview panel exists and has correct width
            assert gui.preview_panel.width() == 150
            
            app.quit()
            
    except ImportError as e:
        pytest.skip(f"GUI dependencies not available: {e}")


def test_annotation_gui_creation_headless():
    """Test that AnnotationGUI can be instantiated in headless mode."""
    # Set up headless environment  
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    try:
        from annotation_gui_pyqt import AnnotationGUI, create_app
        import annotation_pipeline as ap
        
        # Mock the pipeline configuration
        with patch('annotation_gui_pyqt.ap._ensure_cfg') as mock_cfg:
            mock_config = Mock()
            mock_config.preview_scaling = 0.5
            mock_cfg.return_value = mock_config
            
            # Create app and GUI
            app = create_app()
            
            gui = AnnotationGUI()
            
            # Basic checks
            assert gui.windowTitle() == "Annotation Pipeline"
            assert gui.video_path == ""
            assert gui.output_dir == ""
            
            # Check that preview panel exists and has correct width
            assert gui.preview_panel.width() == 300
            
            app.quit()
            
    except ImportError as e:
        pytest.skip(f"Annotation GUI dependencies not available: {e}")


def test_gui_widgets_functionality():
    """Test basic widget functionality in headless mode."""
    # Set up headless environment
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    try:
        from GUI_PyQt import BearVisionGUI, create_app
        
        with patch('GUI_PyQt.ConfigurationHandler') as mock_config:
            mock_config.get_configuration.return_value = None
            
            app = create_app()
            mock_app_ref = Mock()
            
            gui = BearVisionGUI(mock_app_ref)
            
            # Test status updates
            gui.status_label.setText("Test Status")
            assert gui.status_label.text() == "Test Status"
            
            # Test text entry
            gui.video_folder_entry.setText("/new/path")
            assert gui.video_folder_entry.text() == "/new/path"
            
            # Test button enablement
            gui.run_button.setEnabled(False)
            assert not gui.run_button.isEnabled()
            
            gui.run_button.setEnabled(True)
            assert gui.run_button.isEnabled()
            
            app.quit()
            
    except ImportError as e:
        pytest.skip(f"GUI dependencies not available: {e}")


if __name__ == "__main__":
    # Run tests manually for verification
    print("Testing headless GUI creation...")
    
    if not PYSIDE6_AVAILABLE:
        print(f"⚠️  PySide6 dependencies not available: {PYSIDE6_IMPORT_ERROR}")
        print("Tests will be skipped. Install PySide6 and system dependencies to run GUI tests.")
        sys.exit(0)
    
    # Set headless mode
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    try:
        test_create_app_headless()
        print("✓ QApplication creation test passed")
        
        test_bearvision_gui_creation_headless()
        print("✓ BearVision GUI creation test passed")
        
        test_annotation_gui_creation_headless()
        print("✓ Annotation GUI creation test passed")
        
        test_gui_widgets_functionality()
        print("✓ Widget functionality test passed")
        
        print("\nAll headless GUI tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)