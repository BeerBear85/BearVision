"""Configuration classes for the annotation pipeline."""
from dataclasses import dataclass, field
from typing import List, Any


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
        Index of the video frame currently being processed (0-based).
    total_frames: int
        Total number of frames in the video(s) being processed.
    processed_frame_count: int
        Count of frames that have passed quality checks and been processed.
    riders_detected: int
        Count of rider segments detected in the current session.
    """

    last_function: str = ""
    current_frame: int = 0
    total_frames: int = 0
    processed_frame_count: int = 0
    riders_detected: int = 0


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
class GuiConfig:
    """Settings for the annotation GUI interface.

    Purpose
    -------
    Control the visual appearance and sizing of the annotation GUI preview
    elements to allow users to customize the interface to their preferences.

    Attributes
    ----------
    preview_width: int, default ``280``
        Maximum width in pixels for preview image display.
    preview_panel_width: int, default ``300``
        Width in pixels of the preview panel container.
    preview_image_min_height: int, default ``200``
        Minimum height in pixels for the frame preview area.
    trajectory_image_min_height: int, default ``150``
        Minimum height in pixels for the trajectory preview area.
    default_video_path: str, default ``""``
        Default path to video file for GUI initialization.
    default_output_dir: str, default ``""``
        Default output directory for GUI initialization.
    """

    preview_width: int = 280
    preview_panel_width: int = 300
    preview_image_min_height: int = 200
    trajectory_image_min_height: int = 150
    default_video_path: str = ""
    default_output_dir: str = ""


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
    gui: GuiConfig
        GUI appearance and sizing settings.
    detection_gap_timeout_s: float, default ``3.0``
        Number of seconds without detections after which the current
        trajectory is finalised and a new track is started.
    """

    videos: List[str] = field(default_factory=list)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    yolo: YoloConfig | None = None
    export: ExportConfig = field(default_factory=lambda: ExportConfig(output_dir="dataset"))
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    detection_gap_timeout_s: float = 3.0