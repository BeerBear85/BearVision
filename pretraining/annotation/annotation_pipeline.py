"""Annotation dataset pipeline for video ingestion and labeling."""
import json
from pathlib import Path
from typing import Iterator, List, Dict, Any, Callable

import cv2
import yaml
import numpy as np
from ultralytics import YOLO
import argparse
import sys
from functools import wraps
import logging

# Import configuration classes
from annotation_config import (
    PipelineStatus,
    SamplingConfig, 
    QualityConfig,
    YoloConfig,
    ExportConfig,
    TrajectoryConfig,
    LoggingConfig,
    PipelineConfig,
)

# Import trajectory handling functions
from trajectory_handler import (
    lowpass_filter,
    save_trajectory_image,
    generate_trajectory_during_processing,
    export_segment,
)

# Import gap detection
from gap_detector import GapDetector, GapEvent

# Import DnnHandler for optional ONNX inference
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
try:
    from DnnHandler import DnnHandler
except Exception:  # pragma: no cover - DnnHandler might not be available
    DnnHandler = None


logger = logging.getLogger(__name__)


def setup_logging(config: LoggingConfig):
    """Configure logging with specified level and format.
    
    Purpose
    -------
    Initialize the logging system with the format that includes file and function
    names as required by the issue. This centralizes logging configuration to
    ensure consistent formatting across the pipeline.
    
    Inputs
    ------
    config: LoggingConfig
        Logging configuration specifying level and format.
        
    Outputs
    -------
    None
        Configures the root logger with the specified settings.
    """
    # Check if we're in a test environment by looking for pytest
    import sys
    in_test = 'pytest' in sys.modules
    
    if not in_test:
        # Get root logger and clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set the logging level
        root_logger.setLevel(getattr(logging, config.level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(config.format)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler for debug log in working directory
        file_handler = logging.FileHandler(config.debug_filename)
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    else:
        # In test mode, just set the level on our specific logger to avoid interfering with pytest's caplog
        logger.setLevel(getattr(logging, config.level.upper()))
        # Also set the root logger level if no handlers are configured
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.setLevel(getattr(logging, config.level.upper()))
    
    logger.info("Logging configured with level: %s", config.level)


status = PipelineStatus()


def track(func):
    """Decorator updating :data:`status` with the last function called.

    Purpose
    -------
    Provide lightweight instrumentation by recording each decorated call. This
    avoids invasive logging and keeps overhead minimal for performance.

    Inputs
    ------
    func: Callable
        Function to wrap.

    Outputs
    -------
    Callable
        Wrapped function that updates :data:`status` before executing.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        status.last_function = func.__name__
        return func(*args, **kwargs)

    return wrapper


@track
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



class VidIngest:
    """Video ingestion with adaptive frame sampling."""

    def __init__(self, videos: List[str], config: SamplingConfig):
        """Store video list and sampling configuration.

        Purpose
        -------
        The constructor merely records parameters so that iteration can later
        open files lazily, avoiding upfront validation work.

        Inputs
        ------
        videos: list[str]
            Paths to video files to be processed.
        config: SamplingConfig
            Controls frame skipping behaviour.

        Outputs
        -------
        None
            State is stored on the instance for later use.
        """
        self.videos = videos
        self.config = config

    @track
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield sampled frames from each video.

        Purpose
        -------
        Abstract away the sampling logic so callers can simply iterate over
        frames without worrying about frame rates or file handling.

        Inputs
        ------
        None
            The method uses instance attributes configured at construction.

        Outputs
        -------
        Iterator[dict]
            Each dictionary contains the video path, frame index and image
            array for sampled frames.

        Raises
        ------
        FileNotFoundError
            If a video cannot be opened, likely due to an invalid path or
            unsupported codec.
        """
        for video_path in self.videos:
            logger.debug("Opening video file: %s", video_path)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # Failing silently makes the pipeline appear successful with no output;
                # raising surfaces path or codec issues early for easier debugging.
                logger.error("Failed to open video file: %s", video_path)
                raise FileNotFoundError(f"Cannot open video: {video_path}")
            # Derive a stride based on desired FPS; fall back to a fixed step for
            # robustness when metadata is missing.
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            step = (
                int(max(1, orig_fps / self.config.fps))
                if self.config.fps
                else self.config.step
            )
            logger.debug("Video sampling: original_fps=%.1f, target_fps=%.1f, step=%d", 
                        orig_fps, self.config.fps or (orig_fps / self.config.step), step)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    yield {
                        "video": video_path,
                        "frame_idx": frame_idx,
                        "frame": frame,
                    }
                frame_idx += 1
            cap.release()


class QualityFilter:
    """Filter frames based on blur and luma thresholds."""

    def __init__(self, config: QualityConfig):
        """Record thresholds used during :meth:`check`.

        Purpose
        -------
        Storing the thresholds separately keeps the `check` method lightweight
        and avoids repeatedly accessing the config object.

        Inputs
        ------
        config: QualityConfig
            Contains blur and luma limits.

        Outputs
        -------
        None
            Instance attributes are set for later use.
        """
        self.blur_thr = config.blur
        self.luma_min = config.luma_min
        self.luma_max = config.luma_max

    @track
    def check(self, frame) -> bool:
        """Return ``True`` if a frame passes quality thresholds.

        Purpose
        -------
        Quickly discard frames that are too blurry or too dark/bright, which
        reduces downstream processing and storage.

        Inputs
        ------
        frame: ndarray
            Image in BGR order as produced by OpenCV.

        Outputs
        -------
        bool
            ``True`` when the frame should be kept, ``False`` otherwise.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        luma = gray.mean()
        return blur_val >= self.blur_thr and self.luma_min <= luma <= self.luma_max


class PreLabelYOLO:
    """Run YOLO inference to produce bounding boxes."""

    def __init__(self, config: YoloConfig):
        """Load the chosen YOLO model backend.

        Purpose
        -------
        Dynamically select between an ONNX-based DNN handler and the
        Ultralytics implementation, enabling flexibility without duplicating
        inference code elsewhere.

        Inputs
        ------
        config: YoloConfig
            Source of the weights path and confidence threshold.

        Outputs
        -------
        None
            The model and metadata are stored on ``self`` for later detection.
        """
        self.conf_thr = config.conf_thr
        weights = config.weights
        if weights.endswith(".onnx") and DnnHandler is not None:
            model_name = Path(weights).stem
            self.backend = "dnn"
            logger.info("Loading ONNX model via DnnHandler: %s", model_name)
            self.model = DnnHandler(model_name)
            self.model.init()
            # single-class wakeboarder detection
            self.names = {0: "wakeboarder"}
            logger.debug("DNN backend initialized with single class: wakeboarder")
        else:
            self.backend = "ultralytics"
            logger.info("Loading Ultralytics YOLO model: %s", weights)
            self.model = YOLO(weights)
            try:
                self.names = self.model.names
                logger.debug("YOLO model loaded with %d classes: %s", len(self.names), list(self.names.values()))
            except AttributeError:
                self.names = {}
                logger.warning("Could not retrieve class names from YOLO model")

    @track
    def detect(self, frame) -> List[Dict[str, Any]]:
        """Run inference on ``frame`` and return person detections.

        Purpose
        -------
        Provide a uniform output structure regardless of backend, simplifying
        downstream processing.

        Inputs
        ------
        frame: ndarray
            BGR image to analyze.

        Outputs
        -------
        list[dict]
            Bounding boxes with class, label and confidence fields.
        """
        boxes: List[Dict[str, Any]] = []
        if self.backend == "dnn":
            raw_boxes, confs = self.model.find_person(frame)
            for bb, conf in zip(raw_boxes, confs):
                x, y, w, h = bb
                boxes.append(
                    {
                        "bbox": [x, y, x + w, y + h],
                        "cls": 0,
                        "label": self.names.get(0, "0"),
                        "conf": float(conf),
                    }
                )
        else:
            results = self.model(frame)[0]
            for b in results.boxes:
                # `b.conf` and `b.cls` are numpy arrays with a single element in YOLO
                conf = float(b.conf[0].item())
                if conf < self.conf_thr:
                    continue
                cls_id = int(b.cls[0].item())
                label = self.names.get(cls_id, str(cls_id))
                # Only keep detections of persons to avoid exporting every frame
                if label.lower() != "person" and cls_id != 0:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "cls": cls_id,
                        "label": label,
                        "conf": conf,
                    }
                )
        return boxes


def filter_small_person_boxes(
    boxes: List[Dict[str, Any]],
    image_shape: tuple[int, int],
    min_ratio: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split detections into kept and discarded based on diagonal ratio.

    Purpose
    -------
    Remove exceedingly small person detections that are unlikely to be
    meaningful while leaving other classes untouched.

    Inputs
    ------
    boxes: list[dict]
        Detections produced by the model. Only entries labelled ``person`` are
        considered for filtering.
    image_shape: tuple[int, int]
        Image height and width used to compute the full-image diagonal.
    min_ratio: float
        Minimum allowed ratio between the box diagonal and the image diagonal.

    Outputs
    -------
    tuple[list[dict], list[dict]]
        First list contains boxes that pass the filter, second list holds
        discarded boxes along with their computed ratio for logging.
    """

    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    img_h, img_w = image_shape[:2]
    img_diag = float(np.hypot(img_w, img_h))
    for b in boxes:
        label = b.get("label", "").lower()
        if label != "person" and b.get("cls") != 0:
            # Only filter persons; other classes are kept verbatim.
            kept.append(b)
            continue
        x1, y1, x2, y2 = b["bbox"]
        bb_diag = float(np.hypot(x2 - x1, y2 - y1))
        ratio = bb_diag / img_diag
        if ratio < min_ratio:
            logger.debug(
                "Discarding person detection with diagonal ratio %.6f (threshold %.6f)",
                ratio,
                min_ratio,
            )
            discarded.append({"bbox": b["bbox"], "ratio": ratio})
        else:
            kept.append(b)
    return kept, discarded


# Additional helper to normalize configuration input for run
@track
def _ensure_cfg(cfg: "PipelineConfig | str") -> "PipelineConfig":
    """Return a :class:`PipelineConfig` from a path or object."""
    if isinstance(cfg, PipelineConfig):
        return cfg
    cfg_dict = load_config(cfg)
    sampling = cfg_dict.get("sampling", {})
    if isinstance(sampling, dict):
        sampling = SamplingConfig(**sampling)
    quality = cfg_dict.get("quality", {})
    if isinstance(quality, dict):
        quality = QualityConfig(**quality)
    yolo = cfg_dict.get("yolo")
    if isinstance(yolo, dict):
        yolo = YoloConfig(**yolo)
    export = cfg_dict.get("export", {})
    if isinstance(export, dict):
        export = ExportConfig(**export)
    trajectory = cfg_dict.get("trajectory", {})
    if isinstance(trajectory, dict):
        trajectory = TrajectoryConfig(**trajectory)
    gui = cfg_dict.get("gui", {})
    if isinstance(gui, dict):
        from annotation_config import GuiConfig
        gui = GuiConfig(**gui)
    logging_cfg = cfg_dict.get("logging", {})
    if isinstance(logging_cfg, dict):
        logging_cfg = LoggingConfig(**logging_cfg)
    return PipelineConfig(
        videos=cfg_dict.get("videos", []),
        sampling=sampling,
        quality=quality,
        yolo=yolo,
        export=export,
        trajectory=trajectory,
        gui=gui,
        logging=logging_cfg,
        detection_gap_timeout_s=cfg_dict.get("detection_gap_timeout_s", 3.0),
    )








class DatasetExporter:
    """Save frames, labels and debug information."""

    def __init__(self, config: ExportConfig):
        """Prepare output directories and debug log.

        Purpose
        -------
        Ensure required folders exist before any frames are processed, avoiding
        repetitive checks inside the hot save loop.

        Inputs
        ------
        config: ExportConfig
            Carries the output directory path.

        Outputs
        -------
        None
            Directories are created and a debug file handle is opened.
        """
        self.root = Path(config.output_dir)
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.debug_path = self.root / "debug.jsonl"
        self.debug_file = self.debug_path.open("w", encoding="utf-8")

    @track
    def save(self, item: Dict[str, Any], boxes: List[Dict[str, Any]]):
        """Write one frame and its annotations to disk.

        Purpose
        -------
        Persist data in a YOLO-friendly layout while also logging rich
        information for potential debugging or later analysis.

        Inputs
        ------
        item: dict
            Metadata and pixel data for the frame.
        boxes: list[dict]
            Bounding boxes associated with the frame.

        Outputs
        -------
        None
            Files are written to the export directory and debug log.
        """
        frame = item["frame"]
        idx = item["frame_idx"]
        video_name = Path(item["video"]).stem
        img_name = f"{video_name}_{idx:06d}.jpg"
        lbl_name = f"{video_name}_{idx:06d}.txt"
        logger.debug("Exporting frame %d as %s with %d labels", idx, img_name, len(boxes))
        cv2.imwrite(str(self.img_dir / img_name), frame)
        with open(self.lbl_dir / lbl_name, "w", encoding="utf-8") as f:
            for b in boxes:
                x1, y1, x2, y2 = b["bbox"]
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                f.write(f"{b['cls']} {cx} {cy} {w} {h}\n")
        debug = {
            "image": img_name,
            "labels": boxes,
            "video": item["video"],
            "frame_idx": idx,
        }
        if item.get("discarded_boxes"):
            # Including discarded detections in the log aids post-run auditing
            # without cluttering the exported dataset.
            debug["discarded_boxes"] = item["discarded_boxes"]
        # JSON Lines format allows streaming large debug logs without
        # constructing a massive in-memory structure.
        self.debug_file.write(json.dumps(debug) + "\n")

    def close(self):
        """Flush and close the debug log."""
        self.debug_file.close()
        

class CvatExporter:
    """Save frames and annotations in CVAT XML format."""

    def __init__(self, config: ExportConfig):
        """Create an image directory and buffer annotations.

        Purpose
        -------
        CVAT expects a single XML with all boxes, so we accumulate annotations
        in memory and defer writing until :meth:`close`.

        Inputs
        ------
        config: ExportConfig
            Specifies the output directory.

        Outputs
        -------
        None
            Directories are created and an in-memory list is initialised.
        """
        self.root = Path(config.output_dir)
        self.img_dir = self.root / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.annotations: List[Dict[str, Any]] = []

    @track
    def save(self, item: Dict[str, Any], boxes: List[Dict[str, Any]]):
        """Append annotation metadata for one frame.

        Purpose
        -------
        Images are written immediately to avoid large memory use, while box
        metadata is stored for later XML serialization.

        Inputs
        ------
        item: dict
            Frame metadata including pixel data and identifiers.
        boxes: list[dict]
            Bounding boxes detected in the frame.

        Outputs
        -------
        None
            Updates the internal ``annotations`` list and saves an image.
        """
        frame = item["frame"]
        idx = item["frame_idx"]
        video_name = Path(item["video"]).stem
        img_name = f"{video_name}_{idx:06d}.jpg"
        cv2.imwrite(str(self.img_dir / img_name), frame)
        h, w = frame.shape[:2]
        self.annotations.append({
            "name": img_name,
            "width": w,
            "height": h,
            "boxes": boxes,
        })

    def close(self):
        """Write accumulated annotations to ``annotations.xml``."""
        import xml.etree.ElementTree as ET

        root_el = ET.Element("annotations")
        for i, img in enumerate(self.annotations):
            img_el = ET.SubElement(
                root_el,
                "image",
                id=str(i),
                name=img["name"],
                width=str(img["width"]),
                height=str(img["height"]),
            )
            for b in img["boxes"]:
                x1, y1, x2, y2 = b["bbox"]
                box_el = ET.SubElement(
                    img_el,
                    "box",
                    label=str(b.get("label", b.get("cls", "0"))),
                    xtl=str(x1),
                    ytl=str(y1),
                    xbr=str(x2),
                    ybr=str(y2),
                    occluded="0",
                )
                ET.SubElement(box_el, "attribute", name="confidence").text = str(
                    b.get("conf", 0)
                )

        tree = ET.ElementTree(root_el)
        tree.write(self.root / "annotations.xml", encoding="utf-8", xml_declaration=True)


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


if __name__ == "__main__":
    main()
