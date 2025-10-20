"""Post-processing pipeline for virtual cameraman functionality.

This module orchestrates the complete post-processing workflow including
YOLO detection, trajectory computation, fixed-size bounding box generation,
and JSON metadata export.

Purpose
-------
Implements Part 1 of the Virtual Cameraman feature (GitHub issue #165):
- Run YOLO detections on raw wakeboard videos
- Compute smoothed rider trajectory
- Determine fixed-size bounding box for the clip
- Generate per-frame boxes clamped within frame bounds
- Write JSON metadata file

Classes
-------
PostProcessingPipeline : Main orchestrator for the pipeline

Usage Example
-------------
>>> from PostProcessingConfig import PostProcessingConfig
>>> from PostProcessingPipeline import PostProcessingPipeline
>>>
>>> config = PostProcessingConfig(
...     input_video='raw_clip.mp4',
...     output_json='metadata.json',
...     yolo_model='yolov8n.pt',
...     scaling_factor=1.5
... )
>>>
>>> pipeline = PostProcessingPipeline(config)
>>> result = pipeline.run()
>>> print(f"Processed {result['total_frames']} frames")
>>> print(f"Found {result['num_detections']} detections")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from ultralytics import YOLO

from PostProcessingConfig import PostProcessingConfig
from TrajectoryProcessor import TrajectoryProcessor, Detection, TrajectoryPoint
from BoundingBoxProcessor import BoundingBoxProcessor, FixedBoxSize, BoundingBox


logger = logging.getLogger(__name__)


class PostProcessingPipeline:
    """Main orchestrator for post-processing pipeline.

    This class coordinates YOLO detection, trajectory smoothing, bounding box
    computation, and metadata export to implement the virtual cameraman
    functionality.

    Purpose
    -------
    Provide a simple, high-level API for processing raw wakeboard videos
    into metadata files suitable for automated cropping and editing.

    Parameters
    ----------
    config : PostProcessingConfig
        Configuration with all pipeline parameters

    Attributes
    ----------
    config : PostProcessingConfig
        Pipeline configuration
    yolo_model : YOLO
        Loaded YOLO model for detection
    trajectory_processor : TrajectoryProcessor
        Trajectory smoothing processor
    bbox_processor : BoundingBoxProcessor
        Bounding box computation processor
    frame_width : int
        Video frame width (set during initialization)
    frame_height : int
        Video frame height (set during initialization)
    video_fps : float
        Video frames per second (set during initialization)

    Methods
    -------
    run()
        Execute the complete pipeline
    run_yolo_detections()
        Run YOLO on video and collect detections
    process_detections()
        Convert YOLO results to smoothed trajectory and boxes
    export_metadata(detections, trajectory, boxes, fixed_size)
        Write JSON metadata file

    Usage
    -----
    >>> config = PostProcessingConfig('input.mp4', 'output.json')
    >>> pipeline = PostProcessingPipeline(config)
    >>> result = pipeline.run()
    """

    def __init__(self, config: PostProcessingConfig):
        """Initialize the post-processing pipeline.

        Parameters
        ----------
        config : PostProcessingConfig
            Configuration with pipeline parameters

        Raises
        ------
        ValueError
            If configuration is invalid
        FileNotFoundError
            If input video or model file not found
        """
        self.config = config
        self.config.validate()

        # Initialize video capture to get frame dimensions
        cap = cv2.VideoCapture(str(config.input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {config.input_video}")

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Use video FPS as sample rate if not specified
        if config.sample_rate is None:
            config.sample_rate = self.video_fps / config.frame_skip

        # Initialize YOLO model
        logger.info(f"Loading YOLO model: {config.yolo_model}")
        self.yolo_model = YOLO(str(config.yolo_model))

        # Initialize processors
        self.trajectory_processor = TrajectoryProcessor(
            cutoff_hz=config.cutoff_hz,
            sample_rate=config.sample_rate
        )

        self.bbox_processor = BoundingBoxProcessor(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            scaling_factor=config.scaling_factor,
            preserve_aspect_ratio=config.preserve_aspect_ratio,
            target_aspect_ratio=config.target_aspect_ratio
        )

        logger.info(
            f"Initialized pipeline: {self.frame_width}x{self.frame_height} @ {self.video_fps:.1f} FPS, "
            f"{self.total_frames} frames"
        )

    def run_yolo_detections(self) -> List[Detection]:
        """Run YOLO detection on video and collect detections.

        This method processes the input video frame-by-frame, running YOLO
        person detection and collecting all detections that meet the confidence
        threshold.

        Purpose
        -------
        Extract rider positions from raw video using YOLO object detection.
        Only 'person' class detections above the confidence threshold are kept.

        Returns
        -------
        list of Detection
            All detected rider positions with bounding boxes and confidence scores

        Algorithm
        ---------
        1. Open video file with OpenCV
        2. For each frame (respecting frame_skip):
           a. Run YOLO inference
           b. Filter for 'person' class
           c. Filter by confidence threshold
           d. Extract bounding box center and dimensions
           e. Create Detection object
        3. Return collected detections

        Notes
        -----
        - Only processes every Nth frame according to config.frame_skip
        - Assumes single-person detection (takes highest confidence if multiple)
        - Progress logging every 100 frames if verbose enabled

        Raises
        ------
        RuntimeError
            If video cannot be opened or read fails
        """
        logger.info(f"Running YOLO detection on {self.config.input_video}")
        logger.info(f"Processing every {self.config.frame_skip} frame(s)")

        detections = []
        cap = cv2.VideoCapture(str(self.config.input_video))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.input_video}")

        frame_idx = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames according to frame_skip
                if frame_idx % self.config.frame_skip != 0:
                    frame_idx += 1
                    continue

                # Run YOLO inference
                results = self.yolo_model(
                    frame,
                    conf=self.config.confidence_threshold,
                    device=self.config.device,
                    verbose=False
                )

                # Extract person detections
                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue

                    # Filter for 'person' class (class 0 in COCO)
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id != 0:  # 0 = person in COCO dataset
                            continue

                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])

                        # Calculate center and dimensions
                        w = x2 - x1
                        h = y2 - y1
                        cx = x1 + w / 2
                        cy = y1 + h / 2

                        # Create detection
                        detection = Detection(
                            frame_idx=frame_idx,
                            cx=float(cx),
                            cy=float(cy),
                            w=float(w),
                            h=float(h),
                            confidence=confidence,
                            class_id=cls_id,
                            label='person'
                        )
                        detections.append(detection)

                        # Assume single-person detection, take first/highest confidence
                        break
                    break

                processed_count += 1
                if self.config.verbose and processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} frames, found {len(detections)} detections")

                frame_idx += 1

        finally:
            cap.release()

        logger.info(f"Detection complete: {len(detections)} detections in {processed_count} processed frames")
        return detections

    def process_detections(
        self,
        detections: List[Detection]
    ) -> Tuple[List[TrajectoryPoint], FixedBoxSize, List[Tuple[int, BoundingBox]]]:
        """Process detections into trajectory and bounding boxes.

        This method takes raw YOLO detections and computes:
        1. Smoothed trajectory using cubic spline + low-pass filter
        2. Fixed box size based on max observed size + scaling factor
        3. Per-frame bounding boxes centered on trajectory

        Purpose
        -------
        Transform raw detections into ready-to-use metadata for video cropping.

        Parameters
        ----------
        detections : list of Detection
            Raw YOLO detections from video

        Returns
        -------
        tuple of (trajectory, fixed_size, per_frame_boxes)
            - trajectory: List of smoothed trajectory points
            - fixed_size: Fixed box dimensions for entire clip
            - per_frame_boxes: List of (frame_idx, box) tuples

        Raises
        ------
        ValueError
            If fewer than 2 detections (need at least 2 for trajectory)

        Algorithm
        ---------
        1. Compute smoothed trajectory from detections
        2. Determine fixed box size from trajectory
        3. Generate per-frame boxes centered on trajectory
        4. All boxes are clamped to frame boundaries

        Notes
        -----
        - Trajectory includes interpolated points for every frame in range
        - Fixed box size is same for entire clip
        - Per-frame boxes vary position but not size
        """
        logger.info("Computing smoothed trajectory")

        if len(detections) < 2:
            raise ValueError(f"Need at least 2 detections for trajectory, got {len(detections)}")

        # Compute smoothed trajectory
        trajectory = self.trajectory_processor.compute_smoothed_trajectory(
            detections,
            include_sizes=True
        )

        logger.info(f"Generated trajectory with {len(trajectory)} points")

        # Compute fixed box size
        logger.info("Computing fixed bounding box size")
        fixed_size = self.bbox_processor.compute_fixed_box_size(trajectory)
        logger.info(
            f"Fixed box size: {fixed_size.width:.1f}x{fixed_size.height:.1f} "
            f"(aspect ratio {fixed_size.aspect_ratio:.2f})"
        )

        # Generate per-frame boxes
        logger.info("Generating per-frame bounding boxes")
        per_frame_boxes = self.bbox_processor.generate_per_frame_boxes(
            trajectory,
            fixed_size
        )
        logger.info(f"Generated {len(per_frame_boxes)} per-frame boxes")

        return trajectory, fixed_size, per_frame_boxes

    def export_metadata(
        self,
        detections: List[Detection],
        trajectory: List[TrajectoryPoint],
        per_frame_boxes: List[Tuple[int, BoundingBox]],
        fixed_size: FixedBoxSize
    ) -> None:
        """Export metadata to JSON file.

        This method writes a comprehensive JSON file containing all computed
        metadata including detections, trajectory, fixed box size, and per-frame
        bounding boxes.

        Purpose
        -------
        Create machine-readable metadata file for downstream video processing
        (cropping, rendering, analysis).

        Parameters
        ----------
        detections : list of Detection
            Original YOLO detections
        trajectory : list of TrajectoryPoint
            Smoothed trajectory points
        per_frame_boxes : list of (int, BoundingBox)
            Per-frame bounding boxes
        fixed_size : FixedBoxSize
            Fixed box dimensions

        JSON Format
        -----------
        {
            "metadata": {
                "input_video": "path/to/video.mp4",
                "frame_width": 1920,
                "frame_height": 1080,
                "video_fps": 30.0,
                "total_frames": 300,
                "num_detections": 250,
                "config": {...}
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
                },
                ...
            ],
            "trajectory": [
                {
                    "frame_idx": 0,
                    "x": 960.5,
                    "y": 540.2,
                    "w": 100.3,
                    "h": 200.1
                },
                ...
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
                },
                ...
            ]
        }

        Notes
        -----
        - Output file is pretty-printed with 2-space indentation
        - All numeric values are converted to float for JSON compatibility
        - File is atomically written (write to temp, then rename)
        """
        logger.info(f"Exporting metadata to {self.config.output_json}")

        # Build metadata structure
        metadata = {
            "metadata": {
                "input_video": str(self.config.input_video),
                "output_json": str(self.config.output_json),
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "video_fps": float(self.video_fps),
                "total_frames": self.total_frames,
                "num_detections": len(detections),
                "trajectory_length": len(trajectory),
                "config": self.config.to_dict(),
            },
            "fixed_box_size": fixed_size.to_dict(),
            "detections": [
                {
                    "frame_idx": d.frame_idx,
                    "cx": d.cx,
                    "cy": d.cy,
                    "w": d.w,
                    "h": d.h,
                    "confidence": d.confidence,
                    "bbox": list(d.to_bbox()),
                }
                for d in detections
            ],
            "trajectory": [
                {
                    "frame_idx": p.frame_idx,
                    "x": p.x,
                    "y": p.y,
                    "w": p.w,
                    "h": p.h,
                }
                for p in trajectory
            ],
            "per_frame_boxes": [
                {
                    "frame_idx": frame_idx,
                    "box": box.to_dict(),
                }
                for frame_idx, box in per_frame_boxes
            ],
        }

        # Write to file (atomic write via temp file)
        output_path = Path(self.config.output_json)
        temp_path = output_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            temp_path.replace(output_path)
            logger.info(f"Metadata exported successfully to {output_path}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to export metadata: {e}") from e

    def render_video(
        self,
        per_frame_boxes: List[Tuple[int, BoundingBox]],
        fixed_size: FixedBoxSize
    ) -> None:
        """Render cropped output video using per-frame bounding boxes.

        This method implements Part 2 of the Virtual Cameraman feature by
        cropping the input video according to the computed bounding boxes
        and writing a new output video.

        Purpose
        -------
        Create a "virtual cameraman" video that automatically crops and follows
        the rider throughout the clip, maintaining consistent framing.

        Parameters
        ----------
        per_frame_boxes : list of (int, BoundingBox)
            Per-frame bounding boxes from Part 1 processing
        fixed_size : FixedBoxSize
            Fixed dimensions for output frames

        Raises
        ------
        RuntimeError
            If video rendering fails
        ValueError
            If output_video path is not specified in config

        Algorithm
        ---------
        1. Open input video with OpenCV
        2. Create output video writer with fixed dimensions
        3. For each frame:
           a. Read frame from input
           b. Crop frame according to bounding box
           c. Resize if needed to match fixed size exactly
           d. Write cropped frame to output
        4. Release video handles

        Notes
        -----
        - Output video uses same FPS as input
        - Output codec: MP4V for compatibility
        - Fixed dimensions may differ slightly from bounding box due to integer rounding
        - Progress logged every 100 frames if verbose enabled

        Examples
        --------
        >>> pipeline.render_video(per_frame_boxes, fixed_size)
        """
        if not self.config.output_video:
            raise ValueError("output_video path must be specified for rendering")

        logger.info(f"Rendering cropped video to {self.config.output_video}")

        # Open input video
        cap = cv2.VideoCapture(str(self.config.input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {self.config.input_video}")

        # Create lookup dictionary for per-frame boxes
        box_dict = {frame_idx: box for frame_idx, box in per_frame_boxes}

        # Determine output dimensions (round to nearest integer)
        out_width = int(round(fixed_size.width))
        out_height = int(round(fixed_size.height))

        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(self.config.output_video),
            fourcc,
            self.video_fps,
            (out_width, out_height)
        )

        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {self.config.output_video}")

        try:
            frame_idx = 0
            frames_written = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get bounding box for this frame
                if frame_idx in box_dict:
                    box = box_dict[frame_idx]

                    # Extract bounding box coordinates (ensure integer and within bounds)
                    x1 = int(max(0, box.x1))
                    y1 = int(max(0, box.y1))
                    x2 = int(min(self.frame_width, box.x2))
                    y2 = int(min(self.frame_height, box.y2))

                    # Crop frame
                    cropped = frame[y1:y2, x1:x2]

                    # Resize to exact output dimensions if needed
                    if cropped.shape[1] != out_width or cropped.shape[0] != out_height:
                        cropped = cv2.resize(cropped, (out_width, out_height))

                    # Write cropped frame
                    out.write(cropped)
                    frames_written += 1

                    if self.config.verbose and frames_written % 100 == 0:
                        logger.info(f"Rendered {frames_written} frames")

                frame_idx += 1

            logger.info(f"Video rendering complete: {frames_written} frames written")

        finally:
            cap.release()
            out.release()

    def run(self) -> Dict[str, Any]:
        """Execute the complete post-processing pipeline.

        This is the main entry point that orchestrates all pipeline stages:
        1. Run YOLO detections on video
        2. Compute smoothed trajectory
        3. Determine fixed box size
        4. Generate per-frame boxes
        5. Export JSON metadata
        6. (Optional) Render cropped output video

        Purpose
        -------
        Provide a simple one-call API to process a raw video into metadata
        and optionally a cropped output video.

        Returns
        -------
        dict
            Summary statistics with keys:
            - 'total_frames': Total frames in video
            - 'processed_frames': Frames actually processed
            - 'num_detections': Number of detections found
            - 'trajectory_length': Number of trajectory points
            - 'fixed_box_size': Fixed box dimensions
            - 'output_json': Path to output metadata file
            - 'output_video': Path to output video file (if rendered)

        Raises
        ------
        ValueError
            If insufficient detections for trajectory
        RuntimeError
            If video processing or export fails

        Examples
        --------
        >>> config = PostProcessingConfig('input.mp4', 'output.json')
        >>> pipeline = PostProcessingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Success! Processed {result['num_detections']} detections")
        """
        logger.info("=" * 60)
        logger.info("Starting post-processing pipeline")
        logger.info("=" * 60)

        # Stage 1: Run YOLO detections
        detections = self.run_yolo_detections()

        if len(detections) < 2:
            raise ValueError(
                f"Insufficient detections for trajectory computation. "
                f"Found {len(detections)}, need at least 2. "
                f"Try lowering confidence_threshold or using a better YOLO model."
            )

        # Stage 2: Process detections into trajectory and boxes
        trajectory, fixed_size, per_frame_boxes = self.process_detections(detections)

        # Stage 3: Export metadata
        self.export_metadata(detections, trajectory, per_frame_boxes, fixed_size)

        # Stage 4 (Optional): Render cropped video
        if self.config.output_video:
            self.render_video(per_frame_boxes, fixed_size)

        # Build result summary
        result = {
            'total_frames': self.total_frames,
            'processed_frames': self.total_frames // self.config.frame_skip,
            'num_detections': len(detections),
            'trajectory_length': len(trajectory),
            'fixed_box_size': fixed_size.to_dict(),
            'output_json': str(self.config.output_json),
        }

        # Add output video to result if rendered
        if self.config.output_video:
            result['output_video'] = str(self.config.output_video)

        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"  Detections: {result['num_detections']}")
        logger.info(f"  Trajectory points: {result['trajectory_length']}")
        logger.info(f"  Fixed box: {fixed_size.width:.1f}x{fixed_size.height:.1f}")
        logger.info(f"  JSON output: {result['output_json']}")
        if self.config.output_video:
            logger.info(f"  Video output: {result['output_video']}")
        logger.info("=" * 60)

        return result


__all__ = ['PostProcessingPipeline']
