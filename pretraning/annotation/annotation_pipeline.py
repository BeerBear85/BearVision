"""Annotation dataset pipeline for video ingestion and labeling."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator, List, Dict, Any

import cv2
import yaml
from ultralytics import YOLO
import typer


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
        self.model = YOLO(config.weights)
        self.conf_thr = config.conf_thr

    def detect(self, frame) -> List[Dict[str, Any]]:
        results = self.model(frame)[0]
        boxes = []
        for b in results.boxes:
            conf = float(b.conf[0])
            if conf < self.conf_thr:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "cls": int(b.cls[0]),
                "conf": conf,
            })
        return boxes


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
app = typer.Typer(help="Annotation dataset pipeline")
@app.command()
def run(config_path: str):
    """Run the dataset generation pipeline."""
    cfg_dict = load_config(config_path)
    cfg = PipelineConfig(**cfg_dict)

    ingest = VidIngest(cfg.videos, cfg.sampling)
    qf = QualityFilter(cfg.quality)
    yolo = PreLabelYOLO(cfg.yolo)
    exporter = DatasetExporter(cfg.export)
    for item in ingest:
        if not qf.check(item["frame"]):
            continue
        boxes = yolo.detect(item["frame"])
        exporter.save(item, boxes)
    exporter.close()
@app.command()
def preview(config_path: str):
    """Preview the pipeline output with bounding boxes on screen."""
    cfg_dict = load_config(config_path)
    cfg = PipelineConfig(**cfg_dict)
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
if __name__ == "__main__":
    app()
