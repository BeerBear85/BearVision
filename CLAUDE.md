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

# Box upload testing tool (manual Box cloud storage testing)
python box_upload_gui.py
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

#### Desktop Application Components
- **Application.py**: Main orchestrator coordinating all processing steps
- **GUI.py**: Tkinter-based desktop interface
- **ConfigurationHandler**: Manages INI-based configuration files
- **GoProController**: Interfaces with GoPro cameras via open_gopro library
- **DnnHandler**: YOLO model inference wrapper
- **ble_beacon_handler**: Bluetooth Low Energy beacon detection

#### Edge Application Components (Modular Architecture)
- **EdgeStateMachine**: State machine orchestrating edge application lifecycle
- **EdgeSystemCoordinator**: Coordinates GoPro, YOLO, and BLE operations
- **StatusManager**: Centralized callback and status management
- **StreamProcessor**: Video stream processing and YOLO detection pipeline
- **EdgeThreadManager**: Background thread management (BLE, processing, upload)
- **EdgeApplicationConfig**: Configuration management with typed access and defaults

### Configuration System
The application uses INI configuration files with sections for:
- GUI paths and settings
- Motion detection parameters
- GPS matching criteria
- Clip extraction settings
- Cloud storage credentials
- Edge application settings

Main config: `config.ini`, test config: `test/test_config.ini`

### Cloud Storage
Supports both Google Drive and Box for clip upload:

```ini
[WEB_STORIES]
storage_service = google_drive  # or "box"

[STORAGE_COMMON]
secret_key_name = STORAGE_CREDENTIALS_B64
secret_key_name_2 = STORAGE_CREDENTIALS_B64_2

[GOOGLE_DRIVE]
root_folder = bearvisson_files

[BOX]
root_folder = bearvision_files
```

Both `GoogleDriveHandler` and `BoxHandler` provide:
- `upload_file()`, `download_file()`, `delete_file()`, `list_files()`
- Base64-encoded credentials from environment variables
- Lazy connection initialization for testing

### Directory Structure
- `code/modules/`: Core processing modules
- `code/Application/`: Main app and GUI components
- `code/external_modules/`: Third-party libraries (GPX parser, KBeacon)
- `pretraining/`: ML pipeline for YOLO model training
- `tests/`: Unit tests with pytest framework
- `test/`: Integration test data and runner
- `tools/`: External utilities (ffmpeg, gopro2json)

## Edge Device Mode

The edge application uses a modular state machine architecture for robust, maintainable operation on Raspberry Pi devices.

### State Machine Architecture

The `EdgeStateMachine` coordinates the application through five states:

1. **INITIALIZE**: System startup and resource initialization
   - Connect to GoPro camera
   - Start preview stream
   - Initialize BLE logging thread
   - Start background threads (GUI, cloud upload, post-processing)

2. **LOOKING_FOR_WAKEBOARDER**: Active detection mode
   - Enable GoPro Hindsight mode (if configured)
   - Run YOLO detection on preview stream
   - Wait for wakeboarder detection via callback

3. **RECORDING**: Active recording mode
   - Trigger hindsight clip capture
   - Wait for recording duration to elapse
   - Queue clip for post-processing
   - Return to LOOKING_FOR_WAKEBOARDER

4. **ERROR**: Error recovery mode
   - Log error details
   - Attempt automatic restart (up to configured max attempts)
   - Transition to STOPPING if max restarts exceeded

5. **STOPPING**: Graceful shutdown
   - Stop all background threads
   - Disconnect from GoPro
   - Release resources

### Modular Components

The edge system is built from specialized, testable modules:

- **EdgeStateMachine**: Main state machine orchestrating application flow
- **EdgeSystemCoordinator**: Coordinates GoPro/YOLO/BLE hardware operations
- **StatusManager**: Centralized status updates and callback routing
- **StreamProcessor**: Real-time video processing and YOLO inference
- **EdgeThreadManager**: Manages background threads for async operations
- **EdgeApplicationConfig**: INI-based configuration with validation

### Configuration

Edge application settings in `config.ini` under `[EDGE_APPLICATION]`:

```ini
[EDGE_APPLICATION]
yolo_enabled = true                    # Enable YOLO person detection
yolo_model = yolov8n                   # Model size (n/s/m/l/x)
recording_duration = 30.0              # Clip duration in seconds
detection_cooldown = 2.0               # Min seconds between detections
detection_confidence_threshold = 0.5   # YOLO confidence threshold
hindsight_mode_enabled = true          # Use GoPro Hindsight
preview_stream_enabled = true          # Enable preview display
max_error_restarts = 3                 # Auto-restart attempts
error_restart_delay = 2.0              # Delay between restarts
enable_ble_logging = true              # BLE beacon logging
enable_post_processing = true          # Video post-processing
enable_cloud_upload = true             # Cloud upload thread
```

### Running Edge Application

```bash
# Standard edge mode
python code/Application/edge_main.py

# With custom config
python code/Application/edge_main.py --config path/to/config.ini
```

## Architectural Patterns

### Modular Design Principles
The codebase follows these architectural patterns:

1. **State Machine Pattern**: Edge application uses explicit state transitions for predictable behavior
2. **Separation of Concerns**: Specialized modules for distinct responsibilities (status, streaming, coordination)
3. **Callback Architecture**: Event-driven design with callback functions for async operations
4. **Configuration Management**: Centralized config with typed access and validation
5. **Thread Safety**: Lock-based synchronization for state transitions and shared resources

### Best Practices When Contributing

- **Edge Application**: Always work with the modular components (`EdgeStateMachine`, `EdgeSystemCoordinator`, etc.) rather than the old monolithic design
- **Configuration**: Use `EdgeApplicationConfig` for typed config access rather than raw ConfigParser
- **Status Updates**: Route all status/logging through `StatusManager` for consistency
- **State Transitions**: Use `EdgeStateMachine` state methods; avoid direct state manipulation
- **Threading**: Let `EdgeThreadManager` handle background threads; don't create raw threads
- **Testing**: Unit tests use `pytest`; integration tests use `test/run_test.py`

### Code Organization

- **31 modules** in `code/modules/` - prefer editing existing modules over creating new ones
- **Modular architecture** - each module has a single, well-defined responsibility
- **Documentation** - comprehensive docstrings in state machine and coordinator modules
- **Backup files** - `*_original_backup.py` files preserve pre-refactor versions for reference

## Dependencies
Key Python packages (see requirements.txt):
- OpenCV for computer vision
- ultralytics for YOLO models
- bleak for BLE communication
- google-api-python-client for cloud storage
- open_gopro for camera control
- tkinter for GUI (built into Python)