"""Annotation dataset pipeline for video ingestion and labeling."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator, List, Dict, Any, Callable

import cv2
import yaml
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from ultralytics import YOLO
import argparse
import sys
from functools import wraps
import logging

# Import DnnHandler for optional ONNX inference
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
try:
    from DnnHandler import DnnHandler
except Exception:  # pragma: no cover - DnnHandler might not be available
    DnnHandler = None


logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Keep track of pipeline progress for external observers.

    Purpose
    -------
    Expose runtime state so that user interfaces can surface feedback on the
    pipeline's execution without tightly coupling to internal implementation
    details.

    Attributes
    ----------
    last_function: str
        Name of the most recently invoked function in this module.
    current_frame: int
        One-based index of the frame currently being processed.
    total_frames: int
        Total number of frames scheduled for processing in the current run.
    """

    last_function: str = ""
    current_frame: int = 0
    total_frames: int = 0


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


@dataclass
class SamplingConfig:
    """Control how densely frames are sampled from the input videos.

    Purpose
    -------
    Describe the temporal down-sampling strategy so that high frame rate
    inputs do not overwhelm the pipeline.

    Attributes
    ----------
    fps: float | None, default ``None``
        Target frames-per-second rate. If ``None`` the ``step`` parameter is
        used directly.
    step: int, default ``1``
        Fixed stride in frames when ``fps`` is unspecified.
    min_person_bbox_diagonal_ratio: float, default ``0.001``
        Smallest allowable ratio between a person's bounding-box diagonal and
        the image diagonal. Detections below this threshold are discarded to
        reduce noise from distant subjects.
    """

    fps: float | None = None
    step: int = 1
    min_person_bbox_diagonal_ratio: float = 0.001

@dataclass
class QualityConfig:
    """Thresholds used to discard blurry or poorly lit frames.

    Purpose
    -------
    Avoid exporting frames that are unlikely to yield useful training data by
    checking focus (blur) and brightness (luma).

    Attributes
    ----------
    blur: float, default ``100.0``
        Minimum Laplacian variance allowed; lower values are considered blurry.
    luma_min: int, default ``40``
        Lower bound on the mean pixel intensity.
    luma_max: int, default ``220``
        Upper bound on the mean pixel intensity.
    """

    blur: float = 100.0
    luma_min: int = 40
    luma_max: int = 220

@dataclass
class YoloConfig:
    """Configuration for the YOLO pre-labeling stage.

    Purpose
    -------
    Specify model weights and detection sensitivity so the pipeline can
    bootstrap labels automatically.

    Attributes
    ----------
    weights: str
        Path to the YOLO weights file.
    conf_thr: float, default ``0.25``
        Confidence threshold below which detections are ignored.
    """

    weights: str
    conf_thr: float = 0.25

@dataclass
class ExportConfig:
    """Describe how and where annotations should be written.

    Purpose
    -------
    Allow the pipeline to output datasets in multiple formats without
    hard-coding file paths or layout.

    Attributes
    ----------
    output_dir: str
        Destination directory for exported files.
    format: str, default ``"yolo"``
        Output style, e.g. ``"yolo"`` or ``"cvat"``.
    """

    output_dir: str
    format: str = "yolo"

@dataclass
class TrajectoryConfig:
    """Settings for smoothing the estimated motion path.

    Purpose
    -------
    Allow callers to attenuate high-frequency jitter in the trajectory while
    preserving temporal alignment via zero-phase filtering.

    Attributes
    ----------
    cutoff_hz: float, default ``0.0``
        Cutoff frequency of the Butterworth low-pass filter in Hertz. A value
        of ``0.0`` disables filtering.
    """

    cutoff_hz: float = 0.0

@dataclass
class PipelineConfig:
    """Aggregate configuration for a pipeline run.

    Purpose
    -------
    Bundle all subordinate configuration sections so callers only pass a single
    object around.

    Attributes
    ----------
    videos: list[str]
        Paths to input videos.
    sampling: SamplingConfig
        Controls temporal sampling.
    quality: QualityConfig
        Filters out unsuitable frames.
    yolo: YoloConfig | None
        Parameters for the pre-labeling model; ``None`` disables detection.
    export: ExportConfig
        Dataset export options.
    detection_gap_timeout_s: float, default ``3.0``
        Number of seconds without detections after which the current
        trajectory is finalised and a new track is started.
    preview_scaling: float, default ``0.2``
        Factor by which frames are downscaled for live previews. Lower values
        reduce CPU overhead at the cost of detail.
    """

    videos: List[str] = field(default_factory=list)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    yolo: YoloConfig | None = None
    export: ExportConfig = field(default_factory=lambda: ExportConfig(output_dir="dataset"))
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    detection_gap_timeout_s: float = 3.0
    preview_scaling: float = 0.2


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
        """
        for video_path in self.videos:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            # Derive a stride based on desired FPS; fall back to a fixed step for
            # robustness when metadata is missing.
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            step = (
                int(max(1, orig_fps / self.config.fps))
                if self.config.fps
                else self.config.step
            )
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
            self.model = DnnHandler(model_name)
            self.model.init()
            # single-class wakeboarder detection
            self.names = {0: "wakeboarder"}
        else:
            self.backend = "ultralytics"
            self.model = YOLO(weights)
            try:
                self.names = self.model.names
            except AttributeError:
                self.names = {}

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


# Additional helper to normalize configuration input for run/preview
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
    return PipelineConfig(
        videos=cfg_dict.get("videos", []),
        sampling=sampling,
        quality=quality,
        yolo=yolo,
        export=export,
        trajectory=trajectory,
        detection_gap_timeout_s=cfg_dict.get("detection_gap_timeout_s", 3.0),
        preview_scaling=cfg_dict.get("preview_scaling", 0.2),
    )


def _lowpass_filter(seq: List[float], cutoff_hz: float, sample_rate: float) -> List[float]:
    """Apply zero-phase low-pass filtering to a sequence.

    Purpose
    -------
    Suppress high-frequency jitter in the trajectory without introducing
    latency, keeping filtered positions aligned with their original frames.

    Inputs
    ------
    seq: list[float]
        Sequence of samples representing either x or y coordinates.
    cutoff_hz: float
        Desired cutoff frequency in Hertz.
    sample_rate: float
        Sampling rate of ``seq`` in Hertz.

    Outputs
    -------
    list[float]
        Filtered sequence retaining the original length.
    """

    if cutoff_hz <= 0:
        # Users can disable filtering by setting the cutoff to zero.
        return seq
    order = 1  # First order keeps attenuation mild and preserves realism.
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # ``filtfilt`` runs the filter forward and backward, eliminating phase
    # delay so that annotations remain time-aligned.
    return filtfilt(b, a, seq).tolist()


def _export_segment(
    segment_items: List[Dict[str, Any]],
    det_points: List[tuple[int, float, float, float, float, int, str]],
    exporter: "DatasetExporter | CvatExporter",
    cfg: PipelineConfig,
    sample_rate: float,
    track_id: int,
    frame_callback: Callable[[np.ndarray], None] | None = None,
) -> tuple[List[tuple[int, int]], Dict[str, Any] | None]:
    """Interpolate and export one trajectory segment.

    Parameters
    ----------
    segment_items:
        Frames belonging to the current segment in chronological order.
    det_points:
        Detection tuples ``(frame_idx, cx, cy, w, h, cls, label)`` for the
        segment.
    exporter:
        Destination writer.
    cfg:
        Full pipeline configuration.
    sample_rate:
        Effective frames-per-second used for interpolation.
    track_id:
        Identifier assigned to this trajectory.
    frame_callback:
        Optional function invoked with each frame after saving. Allows callers
        to render progress previews without polluting the core processing loop.

    Returns
    -------
    tuple[list[tuple[int, int]], dict | None]
        The smoothed trajectory and the last frame processed, used for
        optional preview rendering.
    """

    trajectory: List[tuple[int, int]] = []
    final_item: Dict[str, Any] | None = None

    if len(det_points) >= 2:
        frames = [p[0] for p in det_points]
        cxs = [p[1] for p in det_points]
        cys = [p[2] for p in det_points]
        ws = [p[3] for p in det_points]
        hs = [p[4] for p in det_points]
        sx = CubicSpline(frames, cxs)
        sy = CubicSpline(frames, cys)
        sw = CubicSpline(frames, ws)
        sh = CubicSpline(frames, hs)
        idx_map = {it["frame_idx"]: it for it in segment_items}
        first = segment_items[0]["frame_idx"]
        last = segment_items[-1]["frame_idx"]
        interp: List[tuple[int, float, float, float, float, float, Dict[str, Any]]] = []
        for fi in range(first, last + 1):
            item = idx_map.get(fi)
            if item is None:
                continue
            cx = float(sx(fi))
            cy = float(sy(fi))
            if item.get("boxes"):
                b = item["boxes"][0]
                x1, y1, x2, y2 = b["bbox"]
                w = x2 - x1
                h = y2 - y1
                conf = b.get("conf", 0.0)
            else:
                w = float(sw(fi))
                h = float(sh(fi))
                conf = 0.0
            interp.append((fi, cx, cy, w, h, conf, item))

        xs = _lowpass_filter([p[1] for p in interp], cfg.trajectory.cutoff_hz, sample_rate)
        ys = _lowpass_filter([p[2] for p in interp], cfg.trajectory.cutoff_hz, sample_rate)

        for (_fi, _cx, _cy, w, h, conf, item), cx, cy in zip(interp, xs, ys):
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            item["boxes"] = [
                {
                    "bbox": [x1, y1, x2, y2],
                    "cls": det_points[0][5],
                    "label": det_points[0][6],
                    "conf": conf,
                    "track_id": track_id,
                }
            ]
            exporter.save(item, item["boxes"])
            trajectory.append((int(cx), int(cy)))
            final_item = item
            status.current_frame += 1
            if frame_callback:
                disp = item["frame"].copy()  # Copy so overlays don't alter saved data.
                for b in item.get("boxes", []):
                    x1, y1, x2, y2 = map(int, b["bbox"])
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame_callback(disp)
    else:
        for item in segment_items:
            if item.get("boxes"):
                item["boxes"][0]["track_id"] = track_id
                exporter.save(item, item["boxes"])
                final_item = item
                status.current_frame += 1
                if frame_callback:
                    disp = item["frame"].copy()  # Avoid mutating frame that gets written.
                    for b in item.get("boxes", []):
                        x1, y1, x2, y2 = map(int, b["bbox"])
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame_callback(disp)

    return trajectory, final_item
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
def run(
    cfg: "PipelineConfig | str",
    show_preview: bool = False,
    frame_callback: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Run the dataset generation pipeline with optional spline interpolation.

    Purpose
    -------
    Export frames and bounding box annotations, estimating missing person
    locations using spline interpolation and optional low-pass filtering so
    that low-quality models receive a smoother training trajectory.

    Inputs
    ------
    cfg: PipelineConfig | str
        Pipeline configuration object or path to a YAML file describing one.
    show_preview: bool, default ``False``
        When ``True`` an OpenCV window appears only after processing completes
        to display the interpolated trajectory. If no trajectory can be
        computed, the last processed frame with its detection is shown instead.
        The window waits for a keypress on the final frame so users can confirm
        the interpolation.
    frame_callback: Callable[[np.ndarray], None] | None, optional
        Function invoked with each processed frame after saving. Enables
        callers such as GUIs to display live previews without modifying the
        core export logic. ``None`` disables callbacks.

    Outputs
    -------
    None
        The function writes image and label files to disk and returns ``None``.
    """
    cfg = _ensure_cfg(cfg)

    ingest = VidIngest(cfg.videos, cfg.sampling)
    qf = QualityFilter(cfg.quality)
    yolo = PreLabelYOLO(cfg.yolo)
    if cfg.export.format.lower() == "cvat":
        exporter = CvatExporter(cfg.export)
    else:
        exporter = DatasetExporter(cfg.export)

    items: List[Dict[str, Any]] = []
    # Collect frames and detections first so we can interpolate across the
    # entire clip. This trades memory for simplicity and deterministic output.
    for item in ingest:
        frame = item["frame"]
        if not qf.check(frame):
            continue
        # Filter detections before storing them. Applying this here ensures that
        # interpolation and subsequent processing operate only on meaningful
        # boxes, avoiding needless work on specks far away in the frame.
        boxes = yolo.detect(frame)
        boxes, discarded = filter_small_person_boxes(
            boxes,
            frame.shape,
            cfg.sampling.min_person_bbox_diagonal_ratio,
        )
        item["boxes"] = boxes
        if discarded:
            item["discarded_boxes"] = discarded
            if not boxes and isinstance(exporter, DatasetExporter):
                # Record discarded detections even when nothing is exported so
                # analysts can audit which frames were skipped and why.
                exporter.debug_file.write(
                    json.dumps(
                        {
                            "image": None,
                            "labels": [],
                            "video": item["video"],
                            "frame_idx": item["frame_idx"],
                            "discarded_boxes": discarded,
                        }
                    )
                    + "\n"
                )
        items.append(item)
    # The total count is known only after ingestion because quality checks can
    # drop frames; we expose it here for progress displays.
    status.total_frames = len(items)
    status.current_frame = 0

    # Compute centers, widths and heights for frames with detections.
    # Each entry stores the frame index and bounding box geometry for frames
    # where a person was detected. We also keep class/label to reuse for
    # interpolated boxes.
    det_points: List[tuple[int, float, float, float, float, int, str]] = []
    for item in items:
        if item["boxes"]:
            b = item["boxes"][0]  # assume single-person detection
            x1, y1, x2, y2 = b["bbox"]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            det_points.append(
                (
                    item["frame_idx"],
                    cx,
                    cy,
                    w,
                    h,
                    b.get("cls", 0),
                    b.get("label", "person"),
                )
            )

    sample_rate = cfg.sampling.fps if cfg.sampling.fps else 30 / cfg.sampling.step
    gap_frames = int(cfg.detection_gap_timeout_s * sample_rate)
    item_map = {it["frame_idx"]: it for it in items}

    segments: List[List[tuple[int, float, float, float, float, int, str]]] = []
    if det_points:
        start = 0
        for i in range(1, len(det_points)):
            if det_points[i][0] - det_points[i - 1][0] > gap_frames:
                segments.append(det_points[start:i])
                start = i
        segments.append(det_points[start:])

    trajectory: List[tuple[int, int]] = []
    preview_item: Dict[str, Any] | None = None
    track_id = 0
    for seg in segments:
        track_id += 1
        end_frame = seg[-1][0] + gap_frames
        seg_items = [item_map[fi] for fi in range(seg[0][0], end_frame + 1) if fi in item_map]
        traj, final_item = _export_segment(
            seg_items, seg, exporter, cfg, sample_rate, track_id, frame_callback
        )
        if traj:
            trajectory = traj
        if final_item is not None:
            preview_item = final_item

    exporter.close()
    if show_preview and preview_item is not None:
        if trajectory:
            disp = preview_item["frame"].copy()
            pts = np.array(trajectory, dtype=int)
            cv2.polylines(disp, [pts], False, (0, 0, 255), 2)
        else:
            disp = preview_item["frame"].copy()
            for b in preview_item.get("boxes", []):
                x1, y1, x2, y2 = map(int, b["bbox"])
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("trajectory", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return None

@track
def preview(cfg: "PipelineConfig | str") -> None:
    """Preview detections without exporting files.

    Purpose
    -------
    Provide a quick visual check of the pipeline configuration by rendering
    bounding boxes directly to the screen.

    Inputs
    ------
    cfg: PipelineConfig | str
        Configuration object or path to one on disk.

    Outputs
    -------
    None
        Opens a window displaying frames until the user quits.
    """
    cfg = _ensure_cfg(cfg)
    ingest = VidIngest(cfg.videos, cfg.sampling)
    qf = QualityFilter(cfg.quality)
    yolo = PreLabelYOLO(cfg.yolo)
    for item in ingest:
        frame = item["frame"]
        if not qf.check(frame):
            continue
        # Preview mirrors the main pipeline's filtering to ensure users see
        # the same detections that would be exported.
        boxes = yolo.detect(frame)
        boxes, _ = filter_small_person_boxes(
            boxes,
            frame.shape,
            cfg.sampling.min_person_bbox_diagonal_ratio,
        )
        for b in boxes:
            x1, y1, x2, y2 = map(int, b["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


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

    preview_parser = subparsers.add_parser(
        "preview", help="Preview the pipeline output"
    )
    preview_parser.add_argument("config_path")

    args = parser.parse_args(argv)

    if args.command == "run":
        run(args.config_path)
    elif args.command == "preview":
        preview(args.config_path)


if __name__ == "__main__":
    main()
