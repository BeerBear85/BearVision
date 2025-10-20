# Virtual Cameraman - Part 1: Trajectory, Fixed Boxes, and Metadata

## Overview

The Virtual Cameraman post-processing pipeline automatically analyzes wakeboard videos to generate metadata for intelligent video cropping. This is Part 1, which focuses on trajectory computation, bounding box generation, and metadata export.

**What it does:**
- Runs YOLO person detection on raw wakeboard videos
- Computes smoothed rider trajectory using cubic spline interpolation and low-pass filtering
- Determines optimal fixed-size bounding box for the entire clip
- Generates per-frame bounding boxes that follow the rider while staying within frame boundaries
- Exports comprehensive JSON metadata for downstream video processing

**What it does NOT do (Part 2):**
- Video cropping/rendering
- Clip acquisition/transfer
- Final video output

## Quick Start

### Installation

Ensure you have the required dependencies:
```bash
pip install opencv-python ultralytics scipy numpy
```

### Basic Usage

```bash
# Process a video with default settings
python run_post_processing.py input.mp4 output.json

# Use a better YOLO model for improved accuracy
python run_post_processing.py input.mp4 output.json --yolo yolov8m.pt

# Adjust scaling factor for tighter/looser framing
python run_post_processing.py input.mp4 output.json --scaling 1.8

# Speed up processing by skipping frames
python run_post_processing.py input.mp4 output.json --frame-skip 2
```

### Advanced Usage

```bash
# Full control over all parameters
python run_post_processing.py input.mp4 output.json \
    --yolo yolov8l.pt \
    --confidence 0.6 \
    --scaling 2.0 \
    --cutoff 3.0 \
    --aspect-ratio 1.0 \
    --device cuda \
    --verbose
```

## Architecture

The pipeline consists of four main modules:

### 1. TrajectoryProcessor (`code/modules/TrajectoryProcessor.py`)

**Purpose:** Shared library for trajectory smoothing operations

**Key Classes:**
- `Detection`: Represents a YOLO detection with position and size
- `TrajectoryPoint`: Represents a smoothed trajectory point
- `TrajectoryProcessor`: Main API for interpolation and smoothing

**Algorithm:**
1. Cubic spline interpolation of sparse detection points
2. Zero-phase low-pass filtering (Butterworth filter)
3. Frame-by-frame trajectory point generation

**Configuration:**
- `cutoff_hz`: Filter cutoff frequency (default: 2.0 Hz)
- `sample_rate`: Sampling rate in FPS (default: video FPS)
- `filter_order`: Butterworth filter order (default: 1)

### 2. BoundingBoxProcessor (`code/modules/BoundingBoxProcessor.py`)

**Purpose:** Fixed-size bounding box computation and generation

**Key Classes:**
- `BoundingBox`: Rectangle with position and dimensions
- `FixedBoxSize`: Fixed dimensions for entire clip
- `BoundingBoxProcessor`: Main API for box computation

**Algorithm:**
1. Find maximum observed bounding box dimensions across trajectory
2. Take max(width, height) for square-ish box
3. Scale by configurable factor (default: 1.5)
4. Generate per-frame boxes centered on trajectory
5. Clamp boxes to stay within frame boundaries

**Configuration:**
- `scaling_factor`: Box size multiplier (default: 1.5)
- `preserve_aspect_ratio`: Maintain detected aspect ratio (default: True)
- `target_aspect_ratio`: Force specific aspect ratio (optional)

### 3. PostProcessingConfig (`code/modules/PostProcessingConfig.py`)

**Purpose:** Type-safe configuration management

**Key Parameters:**
- **YOLO Detection:**
  - `yolo_model`: Model weights file (yolov8n/s/m/l.pt)
  - `confidence_threshold`: Detection threshold (0.0-1.0)
  - `device`: Inference device (cpu/cuda/mps)

- **Bounding Box:**
  - `scaling_factor`: Box size multiplier (1.0-3.0)
  - `preserve_aspect_ratio`: Maintain aspect ratio
  - `target_aspect_ratio`: Force specific ratio

- **Trajectory Smoothing:**
  - `cutoff_hz`: Low-pass filter cutoff (Hz)
  - `sample_rate`: Sampling rate (FPS)

- **Processing:**
  - `frame_skip`: Process every Nth frame
  - `verbose`: Enable detailed logging

### 4. PostProcessingPipeline (`code/modules/PostProcessingPipeline.py`)

**Purpose:** Main orchestrator coordinating all stages

**Pipeline Stages:**
1. **Detection:** Run YOLO on video frames
2. **Trajectory:** Compute smoothed trajectory from detections
3. **Bounding Box:** Determine fixed size and generate per-frame boxes
4. **Export:** Write JSON metadata file

## Output Format

The pipeline generates a JSON file with this structure:

```json
{
  "metadata": {
    "input_video": "path/to/input.mp4",
    "frame_width": 1920,
    "frame_height": 1080,
    "video_fps": 30.0,
    "total_frames": 300,
    "num_detections": 250,
    "trajectory_length": 300,
    "config": { ... }
  },
  "fixed_box_size": {
    "width": 450.0,
    "height": 600.0,
    "aspect_ratio": 0.75
  },
  "detections": [
    {
      "frame_idx": 0,
      "cx": 960.0,
      "cy": 540.0,
      "w": 100.0,
      "h": 200.0,
      "confidence": 0.92,
      "bbox": [910.0, 440.0, 1010.0, 640.0]
    }
  ],
  "trajectory": [
    {
      "frame_idx": 0,
      "x": 960.5,
      "y": 540.2,
      "w": 100.3,
      "h": 200.1
    }
  ],
  "per_frame_boxes": [
    {
      "frame_idx": 0,
      "box": {
        "x1": 735.5,
        "y1": 240.2,
        "x2": 1185.5,
        "y2": 840.2,
        "width": 450.0,
        "height": 600.0,
        "center_x": 960.5,
        "center_y": 540.2
      }
    }
  ]
}
```

## Configuration Guide

### YOLO Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov8n.pt | Fastest | Lowest | Quick testing, resource-constrained |
| yolov8s.pt | Fast | Good | Balanced performance |
| yolov8m.pt | Medium | Better | **Recommended for production** |
| yolov8l.pt | Slow | Best | Maximum accuracy, powerful hardware |

### Scaling Factor Guidelines

The scaling factor determines how much padding to add around the detected rider:

| Factor | Framing | Use Case |
|--------|---------|----------|
| 1.0 | Tight crop | Minimize video size, rider fills frame |
| 1.2 | Snug | Some breathing room, slight padding |
| 1.5 | Comfortable | **Default**, good balance |
| 1.8 | Loose | More context, rider has space |
| 2.0+ | Very loose | Maximum context, small rider in frame |

### Trajectory Smoothing Parameters

**cutoff_hz** (Low-pass filter cutoff frequency):
- **0.0 Hz:** No filtering (may have jitter)
- **0.5-1.0 Hz:** Very smooth (may miss quick movements)
- **2.0 Hz:** **Default**, balanced smoothness
- **3.0-5.0 Hz:** More responsive (may have slight jitter)

**sample_rate:**
- Typically equals video FPS / frame_skip
- Auto-detected from video if not specified
- Higher rates capture faster motion

### Frame Skip Strategy

Processing every frame provides best accuracy but is slower:

| frame_skip | Processing | Accuracy | Use Case |
|------------|-----------|----------|----------|
| 1 | 100% frames | Highest | **Default**, full quality |
| 2 | 50% frames | Very good | 2x speedup, minimal quality loss |
| 3 | 33% frames | Good | 3x speedup, acceptable for fast motion |
| 5+ | ≤20% frames | Lower | Fast testing only |

## Integration with Existing Code

### Using TrajectoryProcessor in Annotation Pipeline

The trajectory smoothing logic has been refactored into a shared library. To update the annotation pipeline:

```python
from TrajectoryProcessor import TrajectoryProcessor, Detection

# Convert existing detection tuples to Detection objects
detections = [
    Detection(frame_idx=fi, cx=cx, cy=cy, w=w, h=h, confidence=conf)
    for fi, cx, cy, w, h, cls, label in det_points
]

# Create processor with existing config parameters
processor = TrajectoryProcessor(
    cutoff_hz=cfg.trajectory.cutoff_hz,
    sample_rate=sample_rate
)

# Compute smoothed trajectory
trajectory = processor.compute_smoothed_trajectory(detections)

# Extract coordinates for drawing
trajectory_coords = [(int(p.x), int(p.y)) for p in trajectory]
```

### Legacy Compatibility

For backward compatibility, the original `lowpass_filter()` function is still available:

```python
from TrajectoryProcessor import lowpass_filter

filtered = lowpass_filter(sequence, cutoff_hz=2.0, sample_rate=30.0)
```

## Testing

### Unit Tests

Run trajectory and bounding box unit tests:
```bash
pytest tests/unit/test_trajectory_processor.py
pytest tests/unit/test_bounding_box_processor.py
```

### Integration Tests

Use the existing trajectory regression test video:
```bash
# Test with TestMovie3.avi (existing test video)
python run_post_processing.py test/input_video/TestMovie3.avi output.json --verbose

# Verify output JSON
python -c "import json; print(json.load(open('output.json'))['metadata'])"
```

### Test with Synthetic Video

Create a test video with the trajectory gap detection tests:
```bash
pytest tests/integration/test_trajectory_gap_detection.py -v
```

## Performance Considerations

### Memory Usage
- YOLO model: ~50-200 MB depending on variant
- Video frame buffer: ~6 MB per 1080p frame
- Trajectory data: <1 MB for typical clip

### Processing Speed (approximate, 1080p video)

| Configuration | FPS (processed) | Time for 300 frames |
|--------------|-----------------|---------------------|
| yolov8n, CPU, skip=1 | 5-10 FPS | 30-60 seconds |
| yolov8n, CPU, skip=2 | 10-20 FPS | 15-30 seconds |
| yolov8m, CPU, skip=1 | 2-5 FPS | 60-150 seconds |
| yolov8n, CUDA, skip=1 | 30-60 FPS | 5-10 seconds |
| yolov8m, CUDA, skip=1 | 15-30 FPS | 10-20 seconds |

### Optimization Tips

1. **Use GPU if available:** Add `--device cuda` for 5-10x speedup
2. **Skip frames for testing:** Use `--frame-skip 3` during development
3. **Use smaller YOLO model:** yolov8n is sufficient for most cases
4. **Process shorter clips:** Trim video to region of interest first

## Troubleshooting

### "Insufficient detections for trajectory computation"

**Problem:** YOLO found fewer than 2 person detections

**Solutions:**
1. Lower confidence threshold: `--confidence 0.3`
2. Use better YOLO model: `--yolo yolov8m.pt`
3. Check video quality (lighting, focus, rider visibility)
4. Verify rider is visible in frame for extended period

### "Box is larger than frame"

**Problem:** Scaling factor too large for video resolution

**Solutions:**
1. Reduce scaling factor: `--scaling 1.2`
2. Check YOLO detections aren't abnormally large
3. Verify input video resolution is correct

### Trajectory too jittery

**Problem:** Smoothing insufficient for fast motion

**Solutions:**
1. Lower cutoff frequency: `--cutoff 1.0`
2. Process more frames: `--frame-skip 1`
3. Check detection quality (confidence threshold)

### Trajectory too smooth (misses fast movements)

**Problem:** Over-smoothing removes real motion

**Solutions:**
1. Raise cutoff frequency: `--cutoff 3.0` or higher
2. Disable filtering for debugging: `--cutoff 0`
3. Increase sample rate if using frame skip

## Future Enhancements (Part 2)

The following features are planned for Part 2:

1. **Video Cropping/Rendering:**
   - Apply computed bounding boxes to source video
   - Generate cropped output clips
   - Support multiple output formats/resolutions

2. **Advanced Features:**
   - Multi-person tracking
   - Dynamic zoom (varying box size over time)
   - Automatic clip detection and segmentation
   - GPU-accelerated video encoding

3. **Integration:**
   - Direct integration with edge application
   - Real-time processing mode
   - Cloud upload of processed clips

## References

- GitHub Issue #165: Virtual Cameraman - Part 1
- Existing trajectory test: `tests/integration/test_trajectory_gap_detection.py`
- Test video: `test/input_video/TestMovie3.avi`
- YOLO documentation: https://docs.ultralytics.com/

## API Documentation

For detailed API documentation, see module docstrings:
- `TrajectoryProcessor`: Trajectory smoothing API
- `BoundingBoxProcessor`: Bounding box computation API
- `PostProcessingConfig`: Configuration parameters
- `PostProcessingPipeline`: Main pipeline orchestrator

Each module includes comprehensive docstrings with usage examples, algorithm descriptions, and parameter documentation.
