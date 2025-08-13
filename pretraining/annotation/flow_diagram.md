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
│  Collect Items  │ ──► Accumulate processed frames
│                 │     • Real-time gap detection during processing  
│                 │     • Generate trajectories when gap threshold exceeded
│                 │     • Reset tracking for new segments
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Real-time      │ ──► During Processing:
│  Gap Detection  │     • Track detection gaps frame-by-frame
│                 │     • Generate trajectory when gap > timeout
│                 │     • Continue processing remaining segments  
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  End-of-Video   │ ──► After All Frames Processed:
│  Trajectory     │     • Group remaining detections into segments
│  Generation     │     • Split by gaps > timeout threshold
│                 │     • Process any final trajectory segments
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
    trajectory_id: int
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
    detection_gap_timeout_s: float
}
        │
        ▼
Individual Config Objects used by:
• SamplingConfig ──► VidIngest
• QualityConfig ──► QualityFilter  
• YoloConfig ──► PreLabelYOLO
• ExportConfig ──► DatasetExporter/CvatExporter
• TrajectoryConfig ──► lowpass_filter()
• GuiConfig ──► GUI applications (preview dimensions)
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
    last_function: str,
    current_frame: int,    # Actual video frame index being processed
    total_frames: int,     # Total frames in all input videos
    processed_frame_count: int  # Frames passing quality checks
}
        │
        ▼
Available for GUI/external monitoring
```

## Real-time Trajectory Generation Flow

```
[Frame Processing] ──► [Detection Gap Check] ──► Gap > threshold? ──► [Generate Trajectory Image]
         │                      │                      │                          │
         ▼                      │                      │                          ▼
[Accumulate in              ◄─ No                   Yes                  [Save to trajectories/]
 current_segment]                                      │                          │
         │                                             ▼                          ▼
         ▼                                   [export_segment() or                [GUI displays
[Continue Processing]                         generate_trajectory_during_         real-time
                                             processing()]                        trajectory]
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

1. **Real-time Gap Detection**: Gap detection now occurs during frame processing rather than only at the end, enabling immediate trajectory generation.

2. **Dual Trajectory Generation**: 
   - `generate_trajectory_during_processing()` - Creates trajectories when gaps are detected
   - `export_segment()` - Processes final segments at end-of-video

3. **Enhanced Status Tracking**: Pipeline status now distinguishes between actual video frame indices and processed frame counts.

4. **Track ID Assignment**: Each trajectory segment receives a unique track ID for multi-object tracking scenarios.

5. **Modular Exporters**: DatasetExporter (YOLO format with debug logs) vs CvatExporter (XML format, no debug logs).

6. **GUI Integration**: Optional frame callbacks and trajectory image generation support real-time visualization in GUI applications.