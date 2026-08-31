#!/usr/bin/env python3
"""
Box Upload Testing GUI

A minimal cross-platform GUI for manually testing Box upload functionality.
Built with PySide6 and uses the existing BoxHandler module for all Box operations.

Features:
- Show Cloud content (list Box root folder)
- Select local file for upload
- Specify destination folder (default: test_upload)
- Upload with original filename preservation
- Overwrite confirmation dialog
- File size warning (>1MB)
- Comprehensive logging

Usage:
    python box_upload_gui.py
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

# Add code/modules to path so we can import BoxHandler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "modules"))
from BoxHandler import BoxHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("box_upload_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WorkerThread(QThread):
    """Background worker thread for Box operations to prevent UI blocking."""

    finished = Signal(object)  # Success result
    error = Signal(str)  # Error message

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logger.exception(f"Worker thread error: {e}")
            self.error.emit(str(e))


class CloudContentDialog(QDialog):
    """Dialog window showing Box root folder contents."""

    def __init__(self, files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Box Root Folder Contents")
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout()

        # Title label
        title = QLabel("Files in Box Root Folder:")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # Text area for file list
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 10))

        if files:
            content = "\n".join(f"  • {filename}" for filename in files)
            self.text_edit.setText(content)
            logger.info(f"Displayed {len(files)} files in cloud content dialog")
        else:
            self.text_edit.setText("No files found")
            logger.info("No files found in Box root folder")

        layout.addWidget(self.text_edit)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)


class BoxUploadGUI(QMainWindow):
    """Main window for Box upload testing GUI."""

    def __init__(self):
        super().__init__()
        self.box_handler: Optional[BoxHandler] = None
        self.selected_file_path: Optional[str] = None
        self.worker_thread: Optional[WorkerThread] = None

        self.init_ui()
        logger.info("Box Upload GUI initialized")

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Box Upload Testing Tool")
        self.setMinimumSize(600, 400)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Title
        title = QLabel("Box Upload Testing Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Cloud content button
        self.cloud_btn = QPushButton("Show Cloud Content")
        self.cloud_btn.setToolTip("List files in Box root folder")
        self.cloud_btn.clicked.connect(self.show_cloud_content)
        main_layout.addWidget(self.cloud_btn)

        # Separator
        separator1 = QLabel("─" * 80)
        separator1.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(separator1)

        # File selection section
        file_section = QLabel("Local File Selection:")
        file_section.setFont(QFont("Arial", 12, QFont.Bold))
        main_layout.addWidget(file_section)

        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; background: #f9f9f9;")
        file_layout.addWidget(self.file_label, stretch=1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)

        main_layout.addLayout(file_layout)

        # Destination folder section
        dest_section = QLabel("Destination Folder:")
        dest_section.setFont(QFont("Arial", 12, QFont.Bold))
        main_layout.addWidget(dest_section)

        dest_layout = QHBoxLayout()
        dest_label = QLabel("Folder path:")
        dest_layout.addWidget(dest_label)

        self.dest_input = QLineEdit("test_upload")
        self.dest_input.setPlaceholderText("Enter destination folder path")
        self.dest_input.setToolTip("Folder path in Box (relative to root)")
        dest_layout.addWidget(self.dest_input, stretch=1)

        main_layout.addLayout(dest_layout)

        # Upload button
        self.upload_btn = QPushButton("Upload File")
        self.upload_btn.setToolTip("Upload selected file to Box")
        self.upload_btn.clicked.connect(self.upload_file)
        self.upload_btn.setEnabled(False)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        main_layout.addWidget(self.upload_btn)

        # Separator
        separator2 = QLabel("─" * 80)
        separator2.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(separator2)

        # Status section
        status_section = QLabel("Status:")
        status_section.setFont(QFont("Arial", 12, QFont.Bold))
        main_layout.addWidget(status_section)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setFont(QFont("Courier", 9))
        self.status_text.setPlaceholderText("Status messages will appear here...")
        main_layout.addWidget(self.status_text)

        # Info label
        info_label = QLabel("Note: All operations are logged to box_upload_gui.log")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        central_widget.setLayout(main_layout)

        self.update_status("Ready. Click 'Show Cloud Content' to view Box files or 'Browse...' to select a file for upload.")

    def update_status(self, message: str, level: str = "info"):
        """
        Update the status text display and log the message.

        Args:
            message: Status message to display
            level: Log level (info, warning, error)
        """
        # Log the message
        if level == "error":
            logger.error(message)
            prefix = "❌ ERROR: "
        elif level == "warning":
            logger.warning(message)
            prefix = "⚠️  WARNING: "
        else:
            logger.info(message)
            prefix = "ℹ️  "

        # Update GUI
        self.status_text.append(f"{prefix}{message}")

        # Auto-scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.End)
        self.status_text.setTextCursor(cursor)

    def get_box_handler(self) -> BoxHandler:
        """
        Get or create BoxHandler instance.

        Returns:
            BoxHandler instance
        """
        if self.box_handler is None:
            try:
                self.update_status("Initializing Box connection...")
                self.box_handler = BoxHandler()
                logger.info("BoxHandler created successfully")
            except Exception as e:
                error_msg = f"Failed to create BoxHandler: {e}"
                logger.exception(error_msg)
                raise
        return self.box_handler

    def show_cloud_content(self):
        """List files in Box root folder and display in a dialog."""
        self.update_status("Fetching Box root folder contents...")
        self.cloud_btn.setEnabled(False)

        def list_files():
            handler = self.get_box_handler()
            return handler.list_files("")

        self.worker_thread = WorkerThread(list_files)
        self.worker_thread.finished.connect(self._on_cloud_content_loaded)
        self.worker_thread.error.connect(self._on_cloud_content_error)
        self.worker_thread.start()

    def _on_cloud_content_loaded(self, files):
        """Handle successful cloud content retrieval."""
        self.cloud_btn.setEnabled(True)
        self.worker_thread = None

        file_count = len(files)
        self.update_status(f"Successfully retrieved {file_count} file(s) from Box root folder")

        # Show dialog
        dialog = CloudContentDialog(files, self)
        dialog.exec()

    def _on_cloud_content_error(self, error_msg):
        """Handle cloud content retrieval error."""
        self.cloud_btn.setEnabled(True)
        self.worker_thread = None

        self.update_status(f"Failed to retrieve cloud content: {error_msg}", "error")
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to list Box folder contents:\n\n{error_msg}\n\n"
            "Please check your Box credentials and network connection."
        )

    def browse_file(self):
        """Open file dialog to select a local file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Upload",
            "",
            "All Files (*.*)"
        )

        if file_path:
            self.selected_file_path = file_path
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)

            self.file_label.setText(f"{filename} ({size_mb:.2f} MB)")
            self.upload_btn.setEnabled(True)

            self.update_status(f"Selected file: {filename} ({size_mb:.2f} MB)")
            logger.info(f"File selected: {file_path} ({file_size} bytes)")

    def upload_file(self):
        """Upload the selected file to Box."""
        if not self.selected_file_path:
            QMessageBox.warning(self, "No File", "Please select a file first.")
            return

        if not os.path.exists(self.selected_file_path):
            self.update_status("Selected file no longer exists", "error")
            QMessageBox.critical(self, "Error", "The selected file no longer exists.")
            return

        # Check file size and warn if > 1MB
        file_size = os.path.getsize(self.selected_file_path)
        size_mb = file_size / (1024 * 1024)

        if size_mb > 1.0:
            logger.info(f"File size {size_mb:.2f} MB exceeds 1 MB threshold, showing warning")
            reply = QMessageBox.warning(
                self,
                "Large File Warning",
                f"The selected file is {size_mb:.2f} MB in size.\n\n"
                "This may take some time to upload.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                self.update_status("Upload cancelled by user (large file warning)")
                logger.info("Upload cancelled by user due to file size")
                return

        # Get destination folder and build remote path
        dest_folder = self.dest_input.text().strip()
        filename = os.path.basename(self.selected_file_path)

        if dest_folder:
            remote_path = f"{dest_folder}/{filename}"
        else:
            remote_path = filename

        logger.info(f"Preparing to upload {filename} to {remote_path}")

        # Check if file exists and prompt for overwrite
        self.update_status(f"Checking if {remote_path} already exists in Box...")

        try:
            handler = self.get_box_handler()
            handler.connect()

            # Check if file exists
            folder_id = handler._find_folder_id(dest_folder) if dest_folder else handler.root_id
            existing_file = None

            if folder_id:
                existing_file = handler._find_file(folder_id, filename)

            if existing_file:
                logger.info(f"File {remote_path} already exists, prompting for overwrite")
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"The file '{filename}' already exists in '{dest_folder or 'root'}'.\n\n"
                    "Do you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.No:
                    self.update_status("Upload cancelled by user (file exists)")
                    logger.info("Upload cancelled by user - file exists, overwrite declined")
                    return

                overwrite = True
            else:
                overwrite = False

            # Proceed with upload
            self.update_status(f"Uploading {filename} to Box...")
            self.upload_btn.setEnabled(False)
            self.browse_btn.setEnabled(False)
            self.cloud_btn.setEnabled(False)

            def do_upload():
                handler.upload_file(self.selected_file_path, remote_path, overwrite=overwrite)
                return filename, remote_path

            self.worker_thread = WorkerThread(do_upload)
            self.worker_thread.finished.connect(self._on_upload_success)
            self.worker_thread.error.connect(self._on_upload_error)
            self.worker_thread.start()

        except Exception as e:
            error_msg = f"Failed to prepare upload: {e}"
            self.update_status(error_msg, "error")
            logger.exception(error_msg)
            QMessageBox.critical(self, "Error", f"Upload preparation failed:\n\n{error_msg}")

    def _on_upload_success(self, result):
        """Handle successful upload."""
        filename, remote_path = result

        self.upload_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.cloud_btn.setEnabled(True)
        self.worker_thread = None

        self.update_status(f"✅ Successfully uploaded {filename} to {remote_path}")
        logger.info(f"Upload successful: {remote_path}")

        QMessageBox.information(
            self,
            "Upload Successful",
            f"File '{filename}' has been successfully uploaded to:\n\n{remote_path}"
        )

    def _on_upload_error(self, error_msg):
        """Handle upload error."""
        self.upload_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.cloud_btn.setEnabled(True)
        self.worker_thread = None

        self.update_status(f"Upload failed: {error_msg}", "error")
        logger.error(f"Upload failed: {error_msg}")

        QMessageBox.critical(
            self,
            "Upload Failed",
            f"Failed to upload file:\n\n{error_msg}\n\n"
            "Please check the log file for more details."
        )

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Application closing")

        # Clean up worker thread if running
        if self.worker_thread and self.worker_thread.isRunning():
            logger.info("Waiting for background operation to complete...")
            self.worker_thread.wait(5000)  # Wait up to 5 seconds

        event.accept()


def main():
    """Main entry point for the application."""
    logger.info("=" * 80)
    logger.info("Box Upload GUI starting")
    logger.info("=" * 80)

    app = QApplication(sys.argv)
    app.setApplicationName("Box Upload Testing Tool")

    # Set application style
    app.setStyle("Fusion")

    try:
        window = BoxUploadGUI()
        window.show()

        exit_code = app.exec()
        logger.info(f"Application exited with code {exit_code}")
        return exit_code

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"A fatal error occurred:\n\n{e}\n\n"
            "Please check box_upload_gui.log for details."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
