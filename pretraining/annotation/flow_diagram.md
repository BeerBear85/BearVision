# Annotation Pipeline Flow Diagram

This diagram shows the data flow and processing stages in the annotation pipeline.

## High-Level Pipeline Flow

```
[Input Videos] ──► [Configuration] ──► [Pipeline Processing] ──► [Output Dataset]
                                              │
                                              ▼
                                      [Optional Preview/GUI Mode]
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
│  Initialize     │
│  Components:    │
│  • VidIngest    │
│  • QualityFilter│
│  • PreLabelYOLO │
│  • DatasetExporter OR CvatExporter │
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
│  Real-time      │ ──► For each frame with detections:
│  Gap Detection  │     • Track frames_since_last_detection
│  & Processing   │     • Calculate gap_frames = timeout_s * original_video_fps
│                 │     • When gap >= gap_frames and current_det_points exist:
│                 │       ├─ Increment trajectory_id
│                 │       ├─ Call generate_trajectory_during_processing()
│                 │       ├─ Reset current_segment_items and current_det_points
│                 │       └─ Continue with new segment
│                 │     • Accumulate detections in current_det_points
│                 │     • Add frames to current_segment_items
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  End-of-Video   │ ──► After All Frames Processed:
│  Processing     │     • Process any remaining current_det_points as final segment
│                 │     • Group all collected items by trajectory gaps
│                 │     • Split into segments using gap_frames threshold
│                 │     • Call export_segment() for each remaining segment
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
│                 │     • Images (.jpg)
│                 │     • Labels (YOLO .txt OR CVAT .xml)  
│                 │     • Debug log (.jsonl) - DatasetExporter only
│                 │     • Trajectory images (GUI mode) - real-time & end
│                 │     • Track IDs in bounding box annotations
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
    original_video_fps: float
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
• detection_gap_timeout_s ──► Gap detection logic (converted to gap_frames)
```

## Preview Mode Flow (Simplified)

```
[Input Videos] ──► [Frame Ingestion] ──► [Quality Check] ──► [YOLO Detection] ──► [Box Filtering] ──► [Real-time Display]
                                                                                                              │
                                                                                                              ▼
                                                                                                      [OpenCV Window]
                                                                                                      [User presses 'q' to quit]
```

## Status Tracking Flow

```
@track decorator ──► Updates PipelineStatus {
    last_function: str,           # Name of most recently called @track function
    current_frame: int,           # Actual video frame index being processed  
    total_frames: int,            # Total frames in all input videos (from metadata)
    processed_frame_count: int    # Count of frames passing quality checks and processed
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
• Minimal preview                           • Live preview with bounding boxes
                                           • Trajectory visualization
                                           • Status monitoring
```

## Key Architectural Changes

1. **Real-time Gap Detection**: Gap detection occurs frame-by-frame during processing using `frames_since_last_detection` counter compared against `gap_frames` threshold calculated from `detection_gap_timeout_s * original_video_fps`.

2. **Dual Trajectory Generation**: 
   - `generate_trajectory_during_processing()` - Creates trajectory images when gaps are detected during processing
   - `export_segment()` - Processes remaining segments at end-of-video with full interpolation and export

3. **Enhanced Status Tracking**: PipelineStatus tracks both `current_frame` (actual video frame index) and `processed_frame_count` (frames passing quality checks), enabling accurate progress monitoring.

4. **Sequential Track ID Assignment**: Each trajectory segment receives an incrementing `trajectory_id` starting from 1, enabling multi-object tracking and unique trajectory identification.

5. **Modular Exporters**: DatasetExporter (YOLO format with debug.jsonl logs) vs CvatExporter (XML format, no debug logs), selected based on `export.format` configuration.

6. **Frame-by-Frame Processing**: The pipeline processes frames immediately during ingestion rather than collecting all frames first, enabling real-time gap detection and trajectory generation.

7. **Original Video FPS Usage**: Gap detection uses original video FPS rather than sampling rate to ensure time-based thresholds work correctly regardless of sampling configuration.

8. **GUI Integration**: Optional frame callbacks and trajectory image generation with timestamped filenames support real-time visualization in GUI applications.