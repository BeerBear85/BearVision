"""Annotation dataset pipeline for video ingestion and labeling."""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable
import cv2
import numpy as np
import logging
from ultralytics import YOLO

# Import configuration classes
from annotation_config import (
    PipelineConfig,
    LoggingConfig,
    SamplingConfig,
    QualityConfig,
    YoloConfig,
    ExportConfig,
    TrajectoryConfig,
    PipelineStatus as PipelineStatusClass,
)

# Import trajectory handling functions
from trajectory_handler import (
    generate_trajectory_during_processing,
    export_segment,
)

# Import gap detection
from gap_detector import GapDetector

# Import new modular components
from status import status, track
from config_loader import _ensure_cfg, load_config
from processors import VidIngest, QualityFilter, PreLabelYOLO
from exporters import DatasetExporter, CvatExporter
from filters import filter_small_person_boxes
from logging_setup import setup_logging

# Import DnnHandler for testing compatibility
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
try:
    from DnnHandler import DnnHandler
except Exception:  # pragma: no cover - DnnHandler might not be available
    DnnHandler = None

logger = logging.getLogger(__name__)




@track
def _initialize_processors(cfg: PipelineConfig) -> dict:
    """Create and return all processing objects.
    
    Purpose
    -------
    Initialize all pipeline components including video ingestion, quality filtering,
    YOLO detection, gap detection, and export handling. Centralizes component
    creation for better organization and testability.
    
    Inputs
    ------
    cfg: PipelineConfig
        Complete pipeline configuration with all component settings.
        
    Outputs
    -------
    dict
        Dictionary containing initialized processor objects and metadata.
    """
    ingest = VidIngest(cfg.videos, cfg.sampling)
    logger.debug("Initialized video ingestion with %d videos", len(cfg.videos))
    
    qf = QualityFilter(cfg.quality)
    logger.debug("Initialized quality filter: blur_threshold=%.1f, luma_range=[%d,%d]", 
                cfg.quality.blur, cfg.quality.luma_min, cfg.quality.luma_max)
    
    yolo = PreLabelYOLO(cfg.yolo)
    logger.info("Initialized YOLO model: %s (confidence_threshold=%.3f)", cfg.yolo.weights, cfg.yolo.conf_thr)
    
    if cfg.export.format.lower() == "cvat":
        exporter = CvatExporter(cfg.export)
        logger.info("Using CVAT exporter, output directory: %s", cfg.export.output_dir)
    else:
        exporter = DatasetExporter(cfg.export)
        logger.info("Using YOLO dataset exporter, output directory: %s", cfg.export.output_dir)
    
    # Get original video FPS for gap detection
    original_video_fps = 30  # default fallback
    total_frames = 0
    for vid in cfg.videos:
        cap = cv2.VideoCapture(vid)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames += frame_count
            logger.debug("Video %s: %d frames at %.1f fps", vid, frame_count, fps)
            if vid == cfg.videos[0]:  # Use FPS from first video
                original_video_fps = fps
        else:
            logger.warning("Failed to open video: %s", vid)
        cap.release()
    
    gap_detector = GapDetector(cfg.detection_gap_timeout_s, original_video_fps)
    
    # Update status with total frames
    status.total_frames = total_frames
    status.current_frame = 0
    status.processed_frame_count = 0
    logger.info("Processing %d total frames from %d video(s) at %.1f original fps", 
               total_frames, len(cfg.videos), original_video_fps)
    
    return {
        'ingest': ingest,
        'quality_filter': qf,
        'yolo': yolo,
        'exporter': exporter,
        'gap_detector': gap_detector,
        'original_video_fps': original_video_fps
    }


@track
def _process_videos(cfg: PipelineConfig, processors: dict, frame_callback, gui_mode: bool) -> tuple:
    """Main video processing loop with gap detection.
    
    Purpose
    -------
    Process all video frames through quality filtering, YOLO detection, and
    gap detection to identify trajectory segments. Uses the GapDetector to
    manage complex gap timing logic separately from the main processing flow.
    
    Inputs
    ------
    cfg: PipelineConfig
        Pipeline configuration with processing parameters.
    processors: dict
        Dictionary containing initialized processing objects.
    frame_callback: Callable | None
        Optional callback for frame updates during processing.
    gui_mode: bool
        Whether running in GUI mode for preview handling.
        
    Outputs
    -------
    tuple
        Returns (items, segments) where items contains all processed frames
        and segments contains trajectory data for export.
    """
    ingest = processors['ingest']
    qf = processors['quality_filter'] 
    yolo = processors['yolo']
    gap_detector = processors['gap_detector']
    original_video_fps = processors['original_video_fps']
    
    items: List[Dict[str, Any]] = []
    
    # Track current trajectory segment for real-time gap detection
    current_segment_items: List[Dict[str, Any]] = []
    current_det_points: List[tuple[int, float, float, float, float, int, str]] = []
    
    # Save all detection points and segments for export phase
    all_det_points: List[tuple[int, float, float, float, float, int, str]] = []
    all_segments: List[Dict[str, Any]] = []
    
    # Trajectory ID counter
    trajectory_id = 0
    
    sample_rate = cfg.sampling.fps if cfg.sampling.fps else 30 / cfg.sampling.step
    
    logger.info("Starting frame processing loop")
    for item in ingest:
        frame = item["frame"]
        status.current_frame = item["frame_idx"]
        
        if not qf.check(frame):
            logger.debug("Frame %d failed quality check - skipping", item["frame_idx"])
            if frame_callback and gui_mode:
                disp = item["frame"].copy()
                frame_callback(disp)
            continue
        
        status.processed_frame_count += 1
        
        # YOLO detection and filtering
        boxes = yolo.detect(frame)
        logger.debug("YOLO detection at frame %d: found %d raw detections", item["frame_idx"], len(boxes))
        boxes, discarded = filter_small_person_boxes(
            boxes, frame.shape, cfg.sampling.min_person_bbox_diagonal_ratio
        )
        logger.debug("After filtering at frame %d: kept %d detections, discarded %d", 
                    item["frame_idx"], len(boxes), len(discarded))
        
        item["boxes"] = boxes
        if discarded:
            item["discarded_boxes"] = discarded
            if not boxes and isinstance(processors['exporter'], DatasetExporter):
                # Record discarded detections for audit trail
                processors['exporter'].debug_file.write(
                    json.dumps({
                        "image": None, "labels": [], "video": item["video"],
                        "frame_idx": item["frame_idx"], "discarded_boxes": discarded,
                    }) + "\n"
                )
        
        # Gap detection processing
        current_frame_idx = item["frame_idx"]
        has_detection = bool(boxes)
        
        gap_event = gap_detector.process_frame(current_frame_idx, has_detection, sample_rate)
        
        # Handle segment transitions 
        if gap_event.segment_started and not gap_event.is_first_detection:
            # Save previous segment when starting new one after gap
            if current_det_points:
                trajectory_id += 1
                all_det_points.extend(current_det_points)
                end_frame = current_det_points[-1][0] + gap_detector.gap_frames
                image_path, trajectory_points, final_item = generate_trajectory_during_processing(
                    current_segment_items, current_det_points, cfg, trajectory_id, sample_rate
                )
                all_segments.append({
                    'det_points': current_det_points.copy(),
                    'end_frame': end_frame,
                    'trajectory': trajectory_points,
                    'final_item': final_item,
                    'items': current_segment_items.copy()
                })
                status.riders_detected += 1
                logger.info("Saved trajectory (ID: %d). Total riders detected: %d", trajectory_id, status.riders_detected)
            
            # Reset for new segment
            current_segment_items = []
            current_det_points = []
            gap_detector.reset_for_new_segment()
            
        elif gap_event.segment_started and gap_event.is_first_detection:
            # Handle first detection - reset segment to start from first detection frame
            logger.info("First detection at frame %d - resetting segment to start from first detection", current_frame_idx)
            current_segment_items = []
            current_det_points = []
            
        elif gap_event.segment_ended:
            # End current segment due to gap
            last_detection_frame = gap_detector.get_last_detection_frame()
            if current_det_points and last_detection_frame is not None:
                # Trim segment to last detection
                trimmed_items = [item for item in current_segment_items 
                               if item["frame_idx"] <= last_detection_frame]
                logger.debug("Trimming segment from %d to %d frames (removing gap frames after %d)", 
                           len(current_segment_items), len(trimmed_items), last_detection_frame)
                
                trajectory_id += 1
                all_det_points.extend(current_det_points)
                end_frame = current_det_points[-1][0] + gap_detector.gap_frames
                image_path, trajectory_points, final_item = generate_trajectory_during_processing(
                    trimmed_items, current_det_points, cfg, trajectory_id, sample_rate
                )
                all_segments.append({
                    'det_points': current_det_points.copy(),
                    'end_frame': end_frame,
                    'trajectory': trajectory_points,
                    'final_item': final_item,
                    'items': trimmed_items.copy()
                })
                status.riders_detected += 1
                logger.info("Total riders detected after gap: %d", status.riders_detected)
            
            # Reset for potential new segment
            current_segment_items = []
            current_det_points = []
            gap_detector.reset_for_new_segment()
        
        # Add detection to current segment if present
        if has_detection:
            b = boxes[0]  # assume single-person detection
            x1, y1, x2, y2 = b["bbox"]
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            logger.debug("Adding detection at frame %d: bbox=[%.1f,%.1f,%.1f,%.1f], center=[%.1f,%.1f]", 
                        current_frame_idx, x1, y1, x2, y2, cx, cy)
            current_det_points.append((current_frame_idx, cx, cy, w, h, 
                                     b.get("cls", 0), b.get("label", "person")))
        
        # Add item to current segment and main list
        current_segment_items.append(item)
        items.append(item)
        
        # Frame callback for preview
        if frame_callback:
            disp = item["frame"].copy()
            if gui_mode:
                for b in item.get("boxes", []):
                    x1, y1, x2, y2 = map(int, b["bbox"])
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_callback(disp)
    
    # Handle final segment at end of video
    if current_det_points:
        last_detection_frame = gap_detector.get_last_detection_frame() 
        if last_detection_frame is not None:
            trimmed_items = [item for item in current_segment_items 
                           if item["frame_idx"] <= last_detection_frame]
            logger.debug("Trimming final segment from %d to %d frames", 
                       len(current_segment_items), len(trimmed_items))
            
            trajectory_id += 1
            all_det_points.extend(current_det_points)
            end_frame = current_det_points[-1][0] + gap_detector.gap_frames
            image_path, trajectory_points, final_item = generate_trajectory_during_processing(
                trimmed_items, current_det_points, cfg, trajectory_id, sample_rate
            )
            all_segments.append({
                'det_points': current_det_points.copy(),
                'end_frame': end_frame,
                'trajectory': trajectory_points,
                'final_item': final_item,
                'items': trimmed_items.copy()
            })
            status.riders_detected += 1
            logger.info("Final trajectory completed (ID: %d). Total riders: %d", 
                       trajectory_id, status.riders_detected)
    
    return items, all_segments


@track
def _export_results(cfg: PipelineConfig, exporter, segments: List[Dict], items: List[Dict], 
                   frame_callback, gui_mode: bool) -> tuple:
    """Export interpolated frames and trajectories.
    
    Purpose
    -------
    Export all trajectory segments using pre-computed detection points and
    trajectory data from the processing phase. Reuses calculations to avoid
    redundant computation during export.
    
    Inputs
    ------
    cfg: PipelineConfig
        Pipeline configuration for export settings.
    exporter: DatasetExporter | CvatExporter
        Initialized exporter for saving frames and annotations.
    segments: list[dict]
        Pre-computed trajectory segments with detection points.
    items: list[dict] 
        All processed frame items for lookup during export.
    frame_callback: Callable | None
        Optional callback for export progress updates.
    gui_mode: bool
        Whether running in GUI mode.
        
    Outputs
    -------
    tuple
        Returns (trajectory, preview_item) for final preview display.
    """
    sample_rate = cfg.sampling.fps if cfg.sampling.fps else 30 / cfg.sampling.step
    
    trajectory: List[tuple[int, int]] = []
    preview_item: Dict[str, Any] | None = None
    track_id = 0
    
    logger.info("Starting export phase with %d trajectory segments", len(segments))
    
    for seg_data in segments:
        track_id += 1
        seg_trajectory = seg_data.get('trajectory', [])
        seg_final_item = seg_data.get('final_item')
        seg_det_points = seg_data['det_points']
        seg_items = seg_data.get('items', [])
        
        logger.debug("Exporting segment %d with %d detection points", track_id, len(seg_det_points))
        
        traj, final_item = export_segment(
            seg_items, seg_det_points, exporter, cfg, sample_rate, track_id,
            frame_callback, gui_mode, seg_trajectory
        )
        
        if traj:
            trajectory = traj
        if final_item is not None:
            preview_item = final_item
    
    logger.info("Export complete. Processed %d frames, detected %d rider trajectories", 
               status.processed_frame_count, status.riders_detected)
    exporter.close()
    logger.debug("Exporter closed successfully")
    
    return trajectory, preview_item


@track
def _show_preview(show_preview: bool, trajectory: List, preview_item: Dict) -> None:
    """Display final trajectory preview if requested.
    
    Purpose
    -------
    Show OpenCV window with trajectory overlay or detection boxes for
    visual confirmation of processing results. Separated from main
    processing logic for cleaner organization.
    
    Inputs
    ------
    show_preview: bool
        Whether to display preview window.
    trajectory: list
        Trajectory points for overlay drawing.
    preview_item: dict | None
        Final frame item for preview display.
        
    Outputs
    -------
    None
        Displays OpenCV window and waits for keypress.
    """
    if not show_preview or preview_item is None:
        return
        
    if trajectory:
        disp = preview_item["frame"].copy()
        pts = np.array(trajectory, dtype=int)
        cv2.polylines(disp, [pts], False, (0, 0, 255), 2)
        logger.debug("Displaying trajectory preview with %d points", len(trajectory))
    else:
        disp = preview_item["frame"].copy()
        for b in preview_item.get("boxes", []):
            x1, y1, x2, y2 = map(int, b["bbox"])
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        logger.debug("Displaying detection preview with %d boxes", len(preview_item.get("boxes", [])))
    
    cv2.imshow("trajectory", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@track
def run(
    cfg: "PipelineConfig | str",
    show_preview: bool = False,
    frame_callback: Callable[[np.ndarray], None] | None = None,
    gui_mode: bool = False,
) -> None:
    """Run the dataset generation pipeline with optional spline interpolation.

    Purpose
    -------
    Main orchestrator that coordinates all processing stages including video
    ingestion, gap detection, trajectory generation, and export. Refactored
    into focused methods for better maintainability and testability.

    Inputs
    ------
    cfg: PipelineConfig | str
        Pipeline configuration object or path to a YAML file describing one.
    show_preview: bool, default ``False``
        When ``True`` an OpenCV window appears after processing to display
        the interpolated trajectory or final detection frame.
    frame_callback: Callable[[np.ndarray], None] | None, optional
        Function invoked with each processed frame for live preview updates.
    gui_mode: bool, default ``False``
        When ``True``, indicates GUI mode for proper preview handling.

    Outputs
    -------
    None
        The function writes image and label files to disk and returns ``None``.
    """
    cfg = _ensure_cfg(cfg)
    setup_logging(cfg.logging)
    
    logger.info("Starting annotation pipeline with %d video(s)", len(cfg.videos))
    logger.debug("Pipeline configuration: sampling_fps=%.1f, quality_blur=%.1f, gap_timeout=%.1fs", 
                cfg.sampling.fps or 0, cfg.quality.blur, cfg.detection_gap_timeout_s)
    
    # Initialize all processing components
    processors = _initialize_processors(cfg)
    
    # Process videos with gap detection
    items, segments = _process_videos(cfg, processors, frame_callback, gui_mode)
    
    # Export results using pre-computed segments
    trajectory, preview_item = _export_results(
        cfg, processors['exporter'], segments, items, frame_callback, gui_mode
    )
    
    # Show preview if requested
    _show_preview(show_preview, trajectory, preview_item)

    return None


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Purpose
    -------
    Parse CLI arguments and dispatch to the appropriate subcommand.

    Inputs
    ------
    argv: list[str] | None
        Argument list; defaults to :data:`sys.argv` when ``None``.

    Outputs
    -------
    None
        Executes requested subcommand and exits.
    """
    parser = argparse.ArgumentParser(
        description="Annotation dataset pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run the dataset generation pipeline"
    )
    run_parser.add_argument("config_path")

    args = parser.parse_args(argv)

    if args.command == "run":
        run(args.config_path)


# Backward compatibility re-exports
# Re-export the original status instance and classes/functions
PipelineStatus = PipelineStatusClass
from processors import VidIngest, QualityFilter, PreLabelYOLO
from exporters import DatasetExporter, CvatExporter
from filters import filter_small_person_boxes
from config_loader import load_config

# Re-export configuration classes for backward compatibility
SamplingConfig = SamplingConfig
QualityConfig = QualityConfig
YoloConfig = YoloConfig  
ExportConfig = ExportConfig
TrajectoryConfig = TrajectoryConfig
PipelineConfig = PipelineConfig
LoggingConfig = LoggingConfig

# Make key functions available at module level
__all__ = [
    'run', 'setup_logging', 'PipelineStatus', 'status', 'track',
    'VidIngest', 'QualityFilter', 'PreLabelYOLO', 
    'DatasetExporter', 'CvatExporter',
    'filter_small_person_boxes', 'load_config',
    'SamplingConfig', 'QualityConfig', 'YoloConfig',
    'ExportConfig', 'TrajectoryConfig', 'PipelineConfig',
    'LoggingConfig', 'YOLO', 'DnnHandler'
]


if __name__ == "__main__":
    main()