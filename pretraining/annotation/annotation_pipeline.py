"""Annotation dataset pipeline for video ingestion and labeling."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator, List, Dict, Any

import cv2
import yaml
import numpy as np
from scipy.interpolate import CubicSpline
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
    """Run the dataset generation pipeline with optional spline interpolation.

    Purpose
    -------
    Export frames and bounding box annotations, estimating missing person
    locations using spline interpolation so that low-quality models receive a
    smoother training trajectory.

    Inputs
    ------
    cfg: PipelineConfig | str
        Pipeline configuration object or path to a YAML file describing one.
    show_preview: bool, default ``False``
        When ``True`` an OpenCV window shows the bounding boxes for each frame
        and the final trajectory. The window waits for a keypress on the final
        frame so users can confirm the interpolation.

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
        item["boxes"] = yolo.detect(frame)
        items.append(item)

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

    trajectory: List[tuple[int, int]] = []
    if len(det_points) >= 2:
        # Separate frame indices and box geometry for spline fitting.
        frames = [p[0] for p in det_points]
        cxs = [p[1] for p in det_points]
        cys = [p[2] for p in det_points]
        ws = [p[3] for p in det_points]
        hs = [p[4] for p in det_points]
        # Cubic splines provide smooth trajectories even with irregular spacing.
        sx = CubicSpline(frames, cxs)
        sy = CubicSpline(frames, cys)
        sw = CubicSpline(frames, ws)
        sh = CubicSpline(frames, hs)
        first, last = frames[0], frames[-1]
        idx_map = {it["frame_idx"]: it for it in items}
        for fi in range(first, last + 1):
            item = idx_map.get(fi)
            if item is None:
                # If a frame is missing (e.g. skipped due to sampling), we
                # cannot interpolate its image so we simply ignore it.
                continue
            if not item["boxes"]:
                cx = float(sx(fi))
                cy = float(sy(fi))
                w = float(sw(fi))
                h = float(sh(fi))
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                # Confidence is set to zero to signal an estimated box.
                item["boxes"] = [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "cls": det_points[0][5],
                        "label": det_points[0][6],
                        "conf": 0.0,
                    }
                ]
            trajectory.append((int(float(sx(fi))), int(float(sy(fi)))))

    for item in items:
        boxes = item.get("boxes", [])
        if boxes:
            if show_preview:
                disp = item["frame"].copy()
                for b in boxes:
                    x1, y1, x2, y2 = map(int, b["bbox"])
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("preview", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            exporter.save(item, boxes)

    exporter.close()
    if show_preview and trajectory:
        # Display the final frame with the full interpolated trajectory so that
        # annotators can visually verify the estimate before continuing.
        final_item = next(
            (it for it in items if it["frame_idx"] == det_points[-1][0]), None
        )
        if final_item is not None:
            disp = final_item["frame"].copy()
            pts = np.array(trajectory, dtype=int)
            cv2.polylines(disp, [pts], False, (0, 0, 255), 2)
            cv2.imshow("trajectory", disp)
            cv2.waitKey(0)  # Wait for user confirmation
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
