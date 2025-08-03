"""Annotation dataset pipeline for video ingestion and labeling."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator, List, Dict, Any

import cv2
import yaml
from ultralytics import YOLO
import argparse
import sys

# Import DnnHandler for optional ONNX inference
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
try:
    from DnnHandler import DnnHandler
except Exception:  # pragma: no cover - DnnHandler might not be available
    DnnHandler = None


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SamplingConfig:
    fps: float | None = None
    step: int = 1

@dataclass
class QualityConfig:
    blur: float = 100.0
    luma_min: int = 40
    luma_max: int = 220

@dataclass
class YoloConfig:
    weights: str
    conf_thr: float = 0.25

@dataclass
class ExportConfig:
    output_dir: str
    format: str = "yolo"

@dataclass
class PipelineConfig:
    videos: List[str] = field(default_factory=list)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    yolo: YoloConfig | None = None
    export: ExportConfig = field(default_factory=lambda: ExportConfig(output_dir="dataset"))


class VidIngest:
    """Video ingestion with adaptive frame sampling."""

    def __init__(self, videos: List[str], config: SamplingConfig):
        self.videos = videos
        self.config = config

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for video_path in self.videos:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            step = int(max(1, orig_fps / self.config.fps)) if self.config.fps else self.config.step
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
        self.blur_thr = config.blur
        self.luma_min = config.luma_min
        self.luma_max = config.luma_max

    def check(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        luma = gray.mean()
        return blur_val >= self.blur_thr and self.luma_min <= luma <= self.luma_max


class PreLabelYOLO:
    """Run YOLO inference to produce bounding boxes."""

    def __init__(self, config: YoloConfig):
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

    def detect(self, frame) -> List[Dict[str, Any]]:
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


# Additional helper to normalize configuration input for run/preview
def _ensure_cfg(cfg: "PipelineConfig | str") -> "PipelineConfig":
    """Return a :class:`PipelineConfig` from a path or object."""
    if isinstance(cfg, PipelineConfig):
        return cfg
    cfg_dict = load_config(cfg)
    return PipelineConfig(**cfg_dict)


class DatasetExporter:
    """Save frames, labels and debug information."""

    def __init__(self, config: ExportConfig):
        self.root = Path(config.output_dir)
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.debug_path = self.root / "debug.jsonl"
        self.debug_file = self.debug_path.open("w", encoding="utf-8")

    def save(self, item: Dict[str, Any], boxes: List[Dict[str, Any]]):
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
        self.debug_file.write(json.dumps(debug) + "\n")
    def close(self):
        self.debug_file.close()


class CvatExporter:
    """Save frames and annotations in CVAT format."""

    def __init__(self, config: ExportConfig):
        self.root = Path(config.output_dir)
        self.img_dir = self.root / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.annotations: List[Dict[str, Any]] = []

    def save(self, item: Dict[str, Any], boxes: List[Dict[str, Any]]):
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
def run(cfg: "PipelineConfig | str", show_preview: bool = False) -> None:
    """Run the dataset generation pipeline.

    Parameters
    ----------
    cfg:
        Either a path to a YAML configuration file or a preconstructed
        :class:`PipelineConfig` instance. This makes it easier to drive the
        pipeline programmatically from other modules such as the GUI.
    show_preview:
        If ``True``, display a window with bounding boxes overlaid on the
        frames while the dataset is exported.
    """
    cfg = _ensure_cfg(cfg)

    ingest = VidIngest(cfg.videos, cfg.sampling)
    qf = QualityFilter(cfg.quality)
    yolo = PreLabelYOLO(cfg.yolo)
    if cfg.export.format.lower() == "cvat":
        exporter = CvatExporter(cfg.export)
    else:
        exporter = DatasetExporter(cfg.export)

    for item in ingest:
        frame = item["frame"]
        if not qf.check(frame):
            continue
        boxes = yolo.detect(frame)

        if show_preview:
            disp = frame.copy()
            for b in boxes:
                x1, y1, x2, y2 = map(int, b["bbox"])
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("preview", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if boxes:
            exporter.save(item, boxes)
        

    exporter.close()
    if show_preview:
        cv2.destroyAllWindows()

def preview(cfg: "PipelineConfig | str") -> None:
    """Preview the pipeline output with bounding boxes on screen.

    The ``cfg`` argument may be a path to a YAML file or a
    :class:`PipelineConfig` object. This mirrors :func:`run` so that callers
    such as the GUI can construct configurations in code.
    """
    cfg = _ensure_cfg(cfg)
    ingest = VidIngest(cfg.videos, cfg.sampling)
    qf = QualityFilter(cfg.quality)
    yolo = PreLabelYOLO(cfg.yolo)
    for item in ingest:
        frame = item["frame"]
        if not qf.check(frame):
            continue
        boxes = yolo.detect(frame)
        for b in boxes:
            x1, y1, x2, y2 = map(int, b["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
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
