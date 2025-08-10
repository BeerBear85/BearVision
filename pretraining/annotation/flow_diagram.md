# Annotation Pipeline Flow Diagram

This diagram shows the data flow and processing stages in the annotation pipeline.

## High-Level Pipeline Flow

```
[Input Videos] ──► [Configuration] ──► [Pipeline Processing] ──► [Output Dataset]
                                              │
                                              ▼
                                      [Optional Preview]
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
│  • Exporter     │
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
│                 │     with detections for trajectory
│                 │     generation
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Trajectory     │ ──► Group detections into segments:
│  Segmentation   │     • Detect gaps > timeout
│                 │     • Split into separate tracks
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
│                 │     • Debug log (.jsonl)
│                 │     • Trajectory images (GUI mode)
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Final Output   │
│                 │
│  YOLO Format:   │     CVAT Format:
│  • images/      │     • images/
│  • labels/      │     • annotations.xml
│  • debug.jsonl  │
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
Final Bounding Boxes with Track IDs
        │
        ▼
Exported Dataset Files
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
    yolo: YoloConfig,
    export: ExportConfig,
    trajectory: TrajectoryConfig,
    detection_gap_timeout_s: float
}
        │
        ▼
Individual Config Objects used by:
• SamplingConfig ──► VidIngest
• QualityConfig ──► QualityFilter  
• YoloConfig ──► PreLabelYOLO
• ExportConfig ──► DatasetExporter/CvatExporter
• TrajectoryConfig ──► _lowpass_filter()
```

## Preview Mode Flow (Simplified)

```
[Input Videos] ──► [Frame Ingestion] ──► [Quality Check] ──► [YOLO Detection] ──► [Real-time Display]
                                                                                          │
                                                                                          ▼
                                                                                  [OpenCV Window]
                                                                                  [User presses 'q' to quit]
```

## Status Tracking Flow

```
@track decorator ──► Updates PipelineStatus {
    last_function: str,
    current_frame: int, 
    total_frames: int,
    processed_frame_count: int
}
        │
        ▼
Available for GUI/external monitoring
```