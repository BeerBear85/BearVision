# Annotation Pipeline Caller Diagram

This diagram shows the calling relationships between classes and functions in the annotation pipeline.

## Main Entry Points

```
main() ────┐
           ├── run()
           └── preview()
```

## Core Pipeline Flow (run() function)

```
run()
├── _ensure_cfg() ────── load_config()
├── VidIngest()
├── QualityFilter()
├── PreLabelYOLO()
├── DatasetExporter() OR CvatExporter()
├── [Frame Processing Loop]
│   ├── VidIngest.__iter__()
│   ├── QualityFilter.check()
│   ├── PreLabelYOLO.detect()
│   └── filter_small_person_boxes()
├── [Trajectory Segmentation]
├── _export_segment() [for each segment]
│   ├── CubicSpline() [scipy]
│   ├── _lowpass_filter()
│   │   └── butter(), filtfilt() [scipy.signal]
│   ├── DatasetExporter.save() OR CvatExporter.save()
│   └── _save_trajectory_image() [if gui_mode]
└── exporter.close()
```

## Preview Flow

```
preview()
├── _ensure_cfg() ────── load_config()
├── VidIngest()
├── QualityFilter()
├── PreLabelYOLO()
└── [Frame Processing Loop]
    ├── VidIngest.__iter__()
    ├── QualityFilter.check()
    ├── PreLabelYOLO.detect()
    └── filter_small_person_boxes()
```

## Class Dependencies

```
VidIngest
├── Uses: SamplingConfig
└── Calls: cv2.VideoCapture()

QualityFilter
├── Uses: QualityConfig
└── Calls: cv2.cvtColor(), cv2.Laplacian()

PreLabelYOLO
├── Uses: YoloConfig
├── Optionally uses: DnnHandler (if ONNX)
├── Optionally uses: YOLO (Ultralytics)
└── Calls: model inference methods

DatasetExporter
├── Uses: ExportConfig
└── Calls: cv2.imwrite(), json.dumps()

CvatExporter  
├── Uses: ExportConfig
└── Calls: cv2.imwrite(), xml.etree.ElementTree
```

## Utility Functions

```
track() ────── [decorator for status tracking]
           └── Updates: PipelineStatus

_ensure_cfg() ────── load_config()
              └── Creates config objects from dict

filter_small_person_boxes() ────── np.hypot()

_lowpass_filter() ────── butter(), filtfilt()

_save_trajectory_image() ────── cv2.polylines(), cv2.rectangle(), cv2.imwrite()
```

## Configuration Classes (in annotation_config.py)

```
PipelineConfig
├── Contains: SamplingConfig
├── Contains: QualityConfig  
├── Contains: YoloConfig
├── Contains: ExportConfig
├── Contains: TrajectoryConfig
└── Contains: detection_gap_timeout_s

PipelineStatus ────── [Global status instance]
```

## External Dependencies

```
OpenCV (cv2) ────── Video processing, image operations
scipy ────── CubicSpline, butter, filtfilt
ultralytics ────── YOLO model
yaml ────── Configuration loading
numpy ────── Array operations
pathlib ────── File path operations
json ────── Debug logging
xml.etree.ElementTree ────── CVAT export format
```