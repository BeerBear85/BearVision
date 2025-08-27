"""Configuration loading utilities for annotation pipeline."""

from typing import Dict, Any
import yaml

# Import configuration classes
from annotation_config import (
    SamplingConfig, 
    QualityConfig,
    YoloConfig,
    ExportConfig,
    TrajectoryConfig,
    LoggingConfig,
    PipelineConfig,
)

# Import status tracking
from status import track


@track
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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