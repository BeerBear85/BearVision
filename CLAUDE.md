# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BearVision is a computer vision system for automatic wakeboard highlight clip generation. It uses BLE beacons, edge cameras, and YOLO person detection to automatically capture, crop, track and upload wakeboard trick clips. The system consists of desktop applications, edge device controllers, and machine learning pipelines.

## Development Commands

### Running the Application
```bash
# Main desktop GUI application
python main.py

# Edge device controller (lightweight detection)
python code/Application/edge_main.py

# Test runner for core functionality
python test/run_test.py

# YOLO training GUI for fine-tuning models
python run_train_gui.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_annotation_pipeline.py

# Run tests with verbose output
pytest -v
```

### Machine Learning Pipeline
```bash
# Annotation GUI for creating training datasets
python pretraining/annotation/annotation_gui.py

# Train YOLOv8 model on custom dataset
python pretraining/train_yolo.py /path/to/dataset --epochs 100 --batch 8 --onnx-out wakeboard.onnx
```

## Architecture Overview

### Core Processing Pipeline
The application follows a modular pipeline architecture with these main stages:

1. **Motion Detection** (`MotionStartDetector`): Identifies potential trick moments in video files
2. **User Matching** (`MotionTimeUserMatching`): Correlates motion events with GPS track data
3. **Clip Extraction** (`FullClipExtractor`, `TrackerClipExtractor`): Creates final video clips
4. **Object Tracking** (`BearTracker`): YOLO-based person detection and tracking
5. **Cloud Upload** (`GoogleDriveHandler`, `BoxHandler`): Uploads processed clips

### Key Components

- **Application.py**: Main orchestrator coordinating all processing steps
- **GUI.py**: Tkinter-based desktop interface
- **ConfigurationHandler**: Manages INI-based configuration files
- **GoProController**: Interfaces with GoPro cameras via open_gopro library
- **DnnHandler**: YOLO model inference wrapper
- **ble_beacon_handler**: Bluetooth Low Energy beacon detection

### Configuration System
The application uses INI configuration files with sections for:
- GUI paths and settings
- Motion detection parameters
- GPS matching criteria  
- Clip extraction settings
- Cloud storage credentials

Main config: `config.ini`, test config: `test/test_config.ini`

### Directory Structure
- `code/modules/`: Core processing modules
- `code/Application/`: Main app and GUI components
- `code/external_modules/`: Third-party libraries (GPX parser, KBeacon)
- `pretraining/`: ML pipeline for YOLO model training
- `tests/`: Unit tests with pytest framework
- `test/`: Integration test data and runner
- `tools/`: External utilities (ffmpeg, gopro2json)

## Edge Device Mode
The `edge_main.py` provides a lightweight controller for Raspberry Pi deployment that:
- Connects to GoPro cameras for preview streams
- Runs motion detection algorithms
- Triggers video capture based on BLE beacon proximity

## Dependencies
Key Python packages (see requirements.txt):
- OpenCV for computer vision
- ultralytics for YOLO models
- bleak for BLE communication
- google-api-python-client for cloud storage
- open_gopro for camera control
- tkinter for GUI (built into Python)