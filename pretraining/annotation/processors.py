"""Core processing classes for annotation pipeline."""

from pathlib import Path
from typing import Iterator, List, Dict, Any
import cv2
import sys
import logging
from ultralytics import YOLO

# Import configuration classes
from annotation_config import SamplingConfig, QualityConfig, YoloConfig

# Import status tracking
from status import track

# Import DnnHandler for optional ONNX inference
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
try:
    from DnnHandler import DnnHandler
except Exception:  # pragma: no cover - DnnHandler might not be available
    DnnHandler = None

logger = logging.getLogger(__name__)


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