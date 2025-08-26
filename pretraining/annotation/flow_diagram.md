# Annotation Pipeline Flow Diagram

This diagram shows the data flow and processing stages in the annotation pipeline with real-time gap detection and enhanced logging.

## High-Level Pipeline Flow

```
[Input Videos] ──► [Configuration] ──► [Logging Setup] ──► [Pipeline Processing] ──► [Output Dataset]
     │                                       │                        │
     ▼                                       ▼                        ▼
[Status Tracking]                    [File + Console]         [Optional Preview/GUI Mode]
                                      Logging
```

## Detailed Processing Flow

```
┌─────────────────┐
│  Input Videos   │
│  + Config YAML  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   load_config() │ ──► Create PipelineConfig object
│   _ensure_cfg()  │     with all sub-configs
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  setup_logging()│ ──► Configure logging system:
│                 │     • File handler (debug.log)
│                 │     • Console handler
│                 │     • Formatted output with file/function names
│                 │     • Skip in test mode (pytest detected)
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Initialize     │
│  Components:    │
│  • VidIngest    │
│  • QualityFilter│
│  • PreLabelYOLO │
│  • DatasetExporter OR CvatExporter │
│  • PipelineStatus tracking │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Frame Ingestion│ ──► For each video:
│  Loop           │     • Open with cv2.VideoCapture
│                 │     • Calculate sampling step
│                 │     • Yield frames at target FPS
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Quality Check  │ ──► For each frame:
│                 │     • Convert to grayscale  
│                 │     • Check blur (Laplacian variance)
│                 │     • Check luma (brightness)
│                 │     • Filter out poor quality frames
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  YOLO Detection │ ──► Run inference:
│                 │     • DnnHandler (ONNX) OR Ultralytics
│                 │     • Extract person bounding boxes
│                 │     • Apply confidence threshold
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Box Filtering  │ ──► Remove tiny detections:
│                 │     • Calculate bbox diagonal
│                 │     • Compare to image diagonal
│                 │     • Filter by min ratio threshold
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Real-time      │ ──► For each frame during processing:
│  Gap Detection  │     • Track frames_since_last_detection counter
│  & Processing   │     • Calculate gap_frames = timeout_s * original_video_fps
│                 │     • When frame has detection:
│                 │       ├─ Reset frames_since_last_detection = 0
│                 │       ├─ Add detection to current_det_points
│                 │       ├─ Add frame to current_segment_items
│                 │       └─ Update status.riders_detected if new segment
│                 │     • When frame has no detection:
│                 │       ├─ Increment frames_since_last_detection by sample_rate
│                 │       ├─ Check if gap_frames threshold reached
│                 │       ├─ If gap detected: generate trajectory for current segment
│                 │       ├─ Trim segment to last detection frame (remove gap frames)
│                 │       └─ Reset segment tracking variables
│                 │     • All frames callback to GUI for live preview
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  End-of-Video   │ ──► After All Frames Processed:
│  Processing     │     • Generate final trajectory for remaining current_det_points
│                 │     • Trim final segment to last detection frame
│                 │     • Call generate_trajectory_during_processing() for final segment
│                 │     • Update final status.riders_detected count
│                 │     • Post-process all collected items for legacy trajectory export
│                 │     • Split historical items into segments using gap_frames threshold
│                 │     • Call export_segment() for any remaining legacy segments
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  For Each       │
│  Trajectory     │
│  Segment:       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Interpolation  │ ──► If >= 2 detection points:
│                 │     • Create CubicSpline for x,y,w,h
│                 │     • Interpolate between keyframes
│                 │     • Fill missing detections
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Smoothing      │ ──► Apply low-pass filter:
│  (Optional)     │     • Butterworth filter
│                 │     • Zero-phase (filtfilt)
│                 │     • Configurable cutoff frequency
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Export         │ ──► Save to disk:
│                 │     • Images (.jpg) with detailed logging
│                 │     • Labels (YOLO .txt OR CVAT .xml)  
│                 │     • Debug log (.jsonl) - DatasetExporter only
│                 │     • Trajectory images with timestamps - real-time & end
│                 │     • Track IDs in bounding box annotations
│                 │     • Comprehensive logging of export operations
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Final Output   │
│                 │
│  YOLO Format:   │     CVAT Format:
│  • images/      │     • images/
│  • labels/      │     • annotations.xml
│  • debug.jsonl  │     • trajectories/ (GUI mode)
│  • trajectories/│
│    (GUI mode)   │
└─────────────────┘
```

## Data Structures Flow

```
Raw Video Frames
        │
        ▼
Frame Dictionary: {
    "video": path,
    "frame_idx": int,
    "frame": ndarray
}
        │
        ▼
Quality Filtered Frames
        │
        ▼  
Frames with Detection Boxes: {
    ...,
    "boxes": [bbox_dict, ...],
    "discarded_boxes": [small_bbox, ...] (optional)
}
        │
        ▼
Real-time Gap Detection: {
    current_segment_items: [frame_dict, ...],
    current_det_points: [(frame_idx, cx, cy, w, h, cls, label), ...],
    last_detection_frame: int | None,
    trajectory_id: int,
    frames_since_last_detection: int,
    gap_frames: int,  # calculated from detection_gap_timeout_s * original_video_fps
    original_video_fps: float,
    is_first_detection: bool,
    last_gap_video_frame_number: int | None,
    first_bbox_after_gap_video_frame_number: int | None,
    sample_rate: float  # for gap increment calculation
}
        │
        ▼
Detection Points List: [
    (frame_idx, cx, cy, w, h, cls, label),
    ...
]
        │
        ▼
Trajectory Segments: [
    [det_point1, det_point2, ...],
    [det_point3, det_point4, ...],
    ...
]
        │
        ▼
Interpolated Trajectories: [
    (frame_idx, cx, cy, w, h, conf, frame_dict),
    ...
]
        │
        ▼
Smoothed Coordinates: [x_smooth], [y_smooth]
        │
        ▼
Final Bounding Boxes with Track IDs: {
    "bbox": [x1, y1, x2, y2],
    "cls": int,
    "label": str, 
    "conf": float,
    "track_id": int
}
        │
        ▼
Exported Dataset Files + Real-time Trajectory Images
```

## Configuration Flow

```
YAML Config File
        │
        ▼
Dict[str, Any]
        │
        ▼
PipelineConfig {
    videos: List[str],
    sampling: SamplingConfig,
    quality: QualityConfig, 
    yolo: YoloConfig | None,
    export: ExportConfig,
    trajectory: TrajectoryConfig,
    gui: GuiConfig,
    logging: LoggingConfig,  # NEW: Logging configuration
    detection_gap_timeout_s: float  # default 3.0 seconds
}
        │
        ▼
Individual Config Objects used by:
• SamplingConfig ──► VidIngest (fps, step, min_person_bbox_diagonal_ratio)
• QualityConfig ──► QualityFilter (blur, luma_min, luma_max)  
• YoloConfig ──► PreLabelYOLO (weights, conf_thr, device)
• ExportConfig ──► DatasetExporter/CvatExporter (output_dir, format)
• TrajectoryConfig ──► lowpass_filter() (cutoff_hz)
• GuiConfig ──► GUI applications (preview_width, preview_panel_width, etc.)
• LoggingConfig ──► setup_logging() (level, format, file/console handlers)
• detection_gap_timeout_s ──► Gap detection logic (converted to gap_frames)
```


## Status Tracking Flow

```
@track decorator ──► Updates PipelineStatus {
    last_function: str,           # Name of most recently called @track function
    current_frame: int,           # Actual video frame index being processed  
    total_frames: int,            # Total frames in all input videos (from metadata)
    processed_frame_count: int,   # Count of frames passing quality checks and processed
    riders_detected: int          # NEW: Count of rider trajectory segments detected
}
        │
        ▼
Available for GUI/external monitoring
```

## Real-time Trajectory Generation Flow

```
[Frame with Detections] ──► [frames_since_last_detection >= gap_frames?] ──► [Generate Trajectory]
         │                                    │                                        │
         ▼                                    │                                        ▼
[Add to current_det_points]              ◄─ No                                [generate_trajectory_during_processing()]
[Add to current_segment_items]               │                                        │
         │                                    │                                        ▼
         ▼                               ────Yes──────────────►          [Save trajectory image to output/trajectories/]
[Continue Processing]                         │                                        │
                                              ▼                                        ▼
                                    [trajectory_id += 1]                      [Reset current_segment_items,
                                    [Reset tracking variables]                  current_det_points]
                                              │                                        │
                                              ▼                                        ▼
                                    [Continue with new segment]             [GUI displays real-time trajectory]
```

## GUI Mode vs. Standard Mode

```
Standard Mode:                              GUI Mode:
• Export files only                         • Export files  
• No trajectory images                      • Generate trajectory images
• No frame callbacks                        • Real-time frame callbacks
• No live visualization                     • Live preview with bounding boxes
                                           • Trajectory visualization
                                           • Status monitoring
```

## Key Architectural Changes

1. **Enhanced Logging System**: 
   - Configurable logging with `LoggingConfig` (level, format)
   - Dual output: file (debug.log) and console handlers
   - Formatted logs with file/function names for debugging
   - Test mode detection to avoid interference with pytest

2. **Real-time Gap Detection with Trimming**: 
   - Gap detection occurs frame-by-frame using `frames_since_last_detection` counter
   - Segment trimming removes gap frames after last detection
   - Immediate trajectory generation when gaps are detected
   - Enhanced boundary tracking with gap frame numbers

3. **Enhanced Status Tracking**: 
   - `PipelineStatus` tracks `riders_detected` count
   - Real-time updates during gap detection
   - Both `current_frame` (actual video frame index) and `processed_frame_count`
   - Progress monitoring for GUI applications

4. **Dual Trajectory Generation**: 
   - `generate_trajectory_during_processing()` - Real-time trajectory images during processing
   - `export_segment()` - Legacy segments at end-of-video with full interpolation
   - Timestamped trajectory images for GUI visualization

5. **Comprehensive Frame Logging**: 
   - Detailed logging at each processing stage (ingestion, quality check, detection, filtering)
   - Debug information for detection counts and filtering results
   - Frame-by-frame status updates with confidence scores and bounding box coordinates

6. **Sequential Track ID Assignment**: Each trajectory segment receives incrementing `trajectory_id`, enabling multi-object tracking and unique identification.

7. **Modular Exporters**: DatasetExporter (YOLO format with debug.jsonl) vs CvatExporter (XML format), selected based on configuration.

8. **Original Video FPS Usage**: Gap detection uses original video FPS rather than sampling rate for accurate time-based thresholds.

9. **GUI Integration**: Frame callbacks for live preview, trajectory visualization, and real-time status updates.