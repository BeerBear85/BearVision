# EdgeApplication Documentation

## Overview

The EdgeApplication is the core component of the BearVision EDGE system for automatic wakeboard detection and video clip generation. It uses a **modular architecture** with specialized components:

- **StatusManager** - Status tracking and callbacks
- **StreamProcessor** - Video processing and YOLO detection
- **EdgeSystemCoordinator** - GoPro and system coordination
- **EdgeStateMachine** - High-level state management
- **EdgeThreadManager** - Background thread management

## System Flow

1. **Initialize** - Load config, YOLO model, BLE handler
2. **Connect GoPro** - Establish camera connection and configure settings
3. **Start Systems** - Begin preview stream, enable hindsight, start detection
4. **Monitor Loop** - Process frames, run YOLO detection every 5th frame
5. **Detection Response** - Trigger hindsight clip when person detected
6. **Reset** - Return to monitoring after 5-second cooldown

## States

- **INITIALIZING** - System startup
- **READY** - Initialized, waiting for commands
- **LOOKING_FOR_WAKEBOARDER** - Active monitoring
- **MOTION_DETECTED** - Person detected, triggering recording
- **RECORDING** - Clip being recorded
- **ERROR/STOPPED** - Error or shutdown states

## API

### Core Methods
- `initialize()` - Initialize all subsystems
- `start_system()` - Complete startup sequence
- `stop_system()` - Graceful shutdown
- `connect_gopro()` - Connect and configure camera
- `start_preview()` - Begin video stream
- `trigger_hindsight_clip()` - Record hindsight clip

## Video Processing

**Pipeline**: UDP Stream → Frame Validation → YOLO Detection (every 5th frame) → Box Drawing → GUI Callback → Recording Response

**Configuration**: 1080p@30fps, Detection every 5th frame, 50% confidence threshold

## Detection & Recording

**YOLO Detection**: YOLOv8 with 50% confidence threshold detects people in video frames
**BLE Beacons**: Bluetooth proximity detection for additional trigger mechanism
**Response**: Both detection types trigger 1-second hindsight clip with 5-second cooldown

## Configuration

**Files**: `config.ini` (main), `test/test_config.ini` (testing)
**Settings**: YOLO model selection, detection thresholds, GoPro settings, BLE parameters

## GUI Integration

**Callbacks**: `status_callback`, `detection_callback`, `log_callback`, `frame_callback`
**Updates**: Real-time status, live video with detection overlays, system logging, detection events

## Performance

**Resources**: ~200MB memory, moderate CPU, 10-15 Mbps network
**Timing**: 6s startup, <100ms detection latency, <200ms recording response
**Threading**: Main GUI, detection worker, BLE scanner, background management

## Troubleshooting

**No Preview**: Check GoPro USB connection, verify UDP stream (172.24.106.51:8554), check firewall
**YOLO Issues**: Verify model in `dnn_models/`, check OpenCV installation, review thresholds
**BLE Problems**: Check Bluetooth adapter, beacon battery/proximity, system permissions

### Debug Tools
```bash
python tools/simple_gopro_test.py        # Test GoPro connection
python tools/test_preview_pipeline.py    # Test complete pipeline
python tools/test_hindsight_changes.py   # Test hindsight functionality
```

## Architecture Changes (v3.0.0)

**Major Refactoring**: Transformed monolithic 1,499-line file into modular architecture:
- **83% size reduction** in main file (1,499 → 314 lines)
- **5 new modules** with focused responsibilities
- **100% backward compatibility** maintained
- **Improved maintainability** and testability

**New Modules**:
- `StatusManager.py` - Status and callback management
- `StreamProcessor.py` - Video processing and YOLO detection
- `EdgeSystemCoordinator.py` - System coordination
- `EdgeStateMachine.py` - State machine management
- `EdgeThreadManager.py` - Background thread management