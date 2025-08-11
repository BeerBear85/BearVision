"""PySide6-based graphical user interface for configuring and running BearVision."""

import sys
import os
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QFrame, QAbstractItemView,
    QFileDialog, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from ConfigurationHandler import ConfigurationHandler
from Enums import ActionOptions
from MotionROISelector import MotionROISelector

logger = logging.getLogger(__name__)

# Command mapping from existing Tkinter GUI
commands = {
    ActionOptions.GENERATE_MOTION_FILES: "Generate motion start files",
    ActionOptions.INIT_USERS: "Initialize users",
    ActionOptions.MATCH_LOCATION_IN_MOTION_FILES: "Match user locations to motion files",
    ActionOptions.GENERATE_FULL_CLIP_OUTPUTS: "Generate full clip output videos",
    ActionOptions.GENERATE_TRACKER_CLIP_OUTPUTS: "Generate tracker clip output videos",
}


class BearVisionGUI(QMainWindow):
    """PySide6 based front end for selecting and running processing steps."""

    def __init__(self, app_ref):
        """Create the GUI and populate it with widgets.

        Parameters
        ----------
        app_ref : Application
            Reference to the core application used to execute actions.
        """
        super().__init__()
        self.app_ref = app_ref
        tmp_options = ConfigurationHandler.get_configuration()
        
        self.setWindowTitle("BearVision - WakeVision")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create main splitter for layout with preview panel
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Welcome label
        self.welcome_label = QLabel("BearVision - WakeVision")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet("background-color: red; color: white;")
        self.welcome_label.setFont(QFont("Helvetica", 20))
        left_layout.addWidget(self.welcome_label)
        
        # Folder selection frame
        folder_frame = QFrame()
        folder_layout = QVBoxLayout(folder_frame)
        
        # Video folder selection
        video_folder_layout = QHBoxLayout()
        self.video_folder_entry = QLineEdit()
        if tmp_options is not None:
            self.video_folder_entry.setText(os.path.abspath(tmp_options["GUI"]["video_path"]))
        self.video_folder_button = QPushButton("Select input video folder")
        self.video_folder_button.clicked.connect(self.set_input_video_folder)
        video_folder_layout.addWidget(self.video_folder_entry, 3)
        video_folder_layout.addWidget(self.video_folder_button, 1)
        folder_layout.addLayout(video_folder_layout)
        
        # User folder selection  
        user_folder_layout = QHBoxLayout()
        self.user_folder_entry = QLineEdit()
        if tmp_options is not None:
            self.user_folder_entry.setText(os.path.abspath(tmp_options["GUI"]["user_path"]))
        self.user_folder_button = QPushButton("Select user base folder")
        self.user_folder_button.clicked.connect(self.set_user_folder)
        user_folder_layout.addWidget(self.user_folder_entry, 3)
        user_folder_layout.addWidget(self.user_folder_button, 1)
        folder_layout.addLayout(user_folder_layout)
        
        left_layout.addWidget(folder_frame)
        
        # Run options list
        self.run_options = QListWidget()
        self.run_options.setSelectionMode(QAbstractItemView.MultiSelection)
        for alias in commands:
            self.run_options.addItem(commands[alias])
        
        # Select all options by default
        for i in range(self.run_options.count()):
            self.run_options.item(i).setSelected(True)
            
        left_layout.addWidget(self.run_options)
        
        # Motion ROI selection frame
        roi_frame = QFrame()
        roi_layout = QHBoxLayout(roi_frame)
        
        self.motion_roi_entry = QLineEdit()
        if tmp_options is not None:
            self.motion_roi_entry.setText(tmp_options["MOTION_DETECTION"]["search_box_dimensions"])
        self.motion_roi_button = QPushButton("Select motion detection ROI")
        self.motion_roi_button.clicked.connect(self.set_motion_detection_roi)
        roi_layout.addWidget(self.motion_roi_entry, 3)
        roi_layout.addWidget(self.motion_roi_button, 1)
        
        left_layout.addWidget(roi_frame)
        
        # Button frame
        button_layout = QHBoxLayout()
        
        self.config_load_button = QPushButton("Load Config")
        self.config_load_button.clicked.connect(self.load_config)
        self.config_load_button.setStyleSheet("background-color: green; color: white;")
        self.config_load_button.setFont(QFont("Helvetica", 20))
        button_layout.addWidget(self.config_load_button)
        
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run)
        self.run_button.setStyleSheet("background-color: green; color: white;")
        self.run_button.setFont(QFont("Helvetica", 20))
        button_layout.addWidget(self.run_button)
        
        left_layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        if tmp_options is None:
            self.status_label.setText("No parameters")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: yellow; color: black;")
        self.status_label.setFont(QFont("Helvetica", 20))
        left_layout.addWidget(self.status_label)
        
        # Add left panel to splitter
        main_splitter.addWidget(left_panel)
        
        # Right panel - Image preview placeholder (150px wide, flexible height)
        self.preview_panel = QFrame()
        self.preview_panel.setFixedWidth(150)
        self.preview_panel.setStyleSheet("background-color: lightgray; border: 1px solid gray;")
        preview_layout = QVBoxLayout(self.preview_panel)
        
        preview_label = QLabel("Image Preview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_layout.addWidget(preview_label)
        
        placeholder_label = QLabel("(Future image content will appear here)")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setWordWrap(True)
        placeholder_label.setStyleSheet("color: gray; font-style: italic;")
        preview_layout.addWidget(placeholder_label)
        
        # Add preview panel to splitter
        main_splitter.addWidget(self.preview_panel)
        
        # Set splitter proportions - main content gets most space, preview gets 150px
        main_splitter.setSizes([650, 150])

    def set_input_video_folder(self, directory_path: str = None) -> None:
        """Ask the user for a video folder and store the choice.

        Parameters
        ----------
        directory_path : str, optional
            Preselected directory. If None a dialog is shown.
        """
        if directory_path is None:
            directory_path = QFileDialog.getExistingDirectory(
                self, 
                "Select input video folder",
                self.video_folder_entry.text()
            )
        if directory_path:
            self.video_folder_entry.setText(os.path.abspath(directory_path))
            logger.info("Setting input video folder to: %s", directory_path)

    def set_user_folder(self, directory_path: str = None) -> None:
        """Ask the user for the user base folder and store it.

        Parameters
        ----------
        directory_path : str, optional  
            Preselected directory. If None a dialog is shown.
        """
        if directory_path is None:
            directory_path = QFileDialog.getExistingDirectory(
                self,
                "Select user base folder", 
                self.user_folder_entry.text()
            )
        if directory_path:
            self.user_folder_entry.setText(os.path.abspath(directory_path))
            logger.info("Setting user folder to: %s", directory_path)

    def set_motion_detection_roi(self) -> None:
        """Launch ROI selector and update the configuration with the chosen region."""
        temp_roi_selector = MotionROISelector()
        tmp_roi = temp_roi_selector.SelectROI(self.video_folder_entry.text())
        tmp_options = ConfigurationHandler.get_configuration()
        if tmp_options:
            self.motion_roi_entry.setText(tmp_options["MOTION_DETECTION"]["search_box_dimensions"])
        logger.info("Setting motion ROI to: %s", tmp_roi)

    def run(self) -> None:
        """Trigger the application to run selected actions."""
        logger.debug("run()")
        tmp_options = ConfigurationHandler.get_configuration()
        if tmp_options is None:
            self.status_label.setText("No parameters")
            return
            
        self.status_label.setText("Busy")
        self.status_label.repaint()  # Force UI update
        
        # Get selected indices
        selected_items = self.run_options.selectedItems()
        selected_indices = []
        for item in selected_items:
            row = self.run_options.row(item)
            selected_indices.append(row)
        
        # Pass GUI selections to the application
        self.app_ref.run(
            self.video_folder_entry.text(), 
            self.user_folder_entry.text(), 
            tuple(selected_indices)
        )
        self.status_label.setText("Ready")

    def load_config(self) -> None:
        """Load a configuration file and update GUI fields."""
        logger.debug("load_config()")
        config_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select configuration file",
            ConfigurationHandler.get_configuration_path(),
            "Config files (*.ini *.cfg);;All files (*.*)"
        )
        
        if config_file:
            tmp_options = ConfigurationHandler.read_config_file(config_file)
            
            # Update file selection boxes and GUI
            self.set_input_video_folder(tmp_options["GUI"]["video_path"])
            self.set_user_folder(tmp_options["GUI"]["user_path"])
            self.status_label.setText("Ready")


def create_app():
    """Create and return a QApplication instance for headless testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if sys.argv else [])
    return app


def main():
    """Entry point for running the GUI standalone."""
    app = create_app()
    
    # Import here to avoid circular imports when used as module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Modules'))
    from Application import Application
    
    ConfigurationHandler.read_last_used_config_file()
    app_instance = Application()
    
    window = BearVisionGUI(app_instance)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()