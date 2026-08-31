"""Configuration management for post-processing pipeline.

This module provides typed configuration classes for the virtual cameraman
post-processing pipeline, with validation and default values.

Purpose
-------
Centralized configuration management with clear documentation and type safety
for all post-processing parameters.

Classes
-------
PostProcessingConfig : Main configuration dataclass

Usage Example
-------------
>>> config = PostProcessingConfig(
...     input_video='raw_clip.mp4',
...     output_json='metadata.json',
...     yolo_model='yolov8n.pt',
...     scaling_factor=1.5
... )
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing pipeline.

    This dataclass holds all parameters needed for the virtual cameraman
    post-processing pipeline, including input/output paths, YOLO settings,
    trajectory smoothing parameters, and bounding box computation settings.

    Purpose
    -------
    Provide a type-safe, well-documented configuration interface with
    sensible defaults and validation.

    Attributes
    ----------
    input_video : str or Path
        Path to input video file for processing
    output_json : str or Path
        Path to output JSON metadata file
    output_video : str or Path, optional
        Path to output cropped video file (Part 2).
        If None, only JSON metadata is generated (Part 1 mode).
        If specified, renders a cropped video using the computed bounding boxes.
    yolo_model : str or Path, default 'yolov8n.pt'
        Path to YOLO model weights file. Options:
        - 'yolov8n.pt': Nano (fastest, least accurate)
        - 'yolov8s.pt': Small (balanced)
        - 'yolov8m.pt': Medium (more accurate)
        - 'yolov8l.pt': Large (slow, most accurate)
    confidence_threshold : float, default 0.5
        YOLO detection confidence threshold (0.0 to 1.0).
        Detections below this threshold are discarded.
    scaling_factor : float, default 1.5
        Multiplicative factor for fixed box size.
        1.0 = tight crop, 1.5 = comfortable padding, 2.0 = loose framing
    preserve_aspect_ratio : bool, default True
        Whether to preserve detected aspect ratio when scaling box
    target_aspect_ratio : float, optional
        Desired aspect ratio (width/height) for output boxes.
        If None, uses aspect ratio of scaled detection box.
        Common values: 1.0 (square), 16/9 (landscape), 9/16 (portrait)
    cutoff_hz : float, default 2.0
        Low-pass filter cutoff frequency in Hz for trajectory smoothing.
        Lower values = smoother (but may lose quick movements)
        Higher values = more responsive (but may have jitter)
        Typical range: 0.5 to 5.0 Hz
        Set to 0.0 to disable filtering
    sample_rate : float, optional
        Effective sampling rate in frames per second.
        If None, uses video FPS from input file.
    frame_skip : int, default 1
        Process every Nth frame (1 = every frame, 2 = every other frame).
        Higher values speed up processing but may miss fast motion.
    device : str, default 'cpu'
        Device for YOLO inference: 'cpu', 'cuda', 'mps' (Apple Silicon)
    verbose : bool, default False
        Enable verbose logging output

    Methods
    -------
    validate()
        Validate configuration parameters and raise ValueError if invalid
    to_dict()
        Convert to dictionary for JSON serialization

    Examples
    --------
    >>> # Minimal configuration (uses defaults)
    >>> config = PostProcessingConfig(
    ...     input_video='clip.mp4',
    ...     output_json='metadata.json'
    ... )
    >>>
    >>> # Custom configuration
    >>> config = PostProcessingConfig(
    ...     input_video='clip.mp4',
    ...     output_json='metadata.json',
    ...     yolo_model='yolov8m.pt',
    ...     scaling_factor=1.8,
    ...     cutoff_hz=3.0,
    ...     confidence_threshold=0.6
    ... )
    >>> config.validate()
    """

    # Required parameters
    input_video: str
    output_json: str

    # Optional output video (Part 2)
    output_video: Optional[str] = None

    # YOLO detection settings
    yolo_model: str = 'yolov8n.pt'
    confidence_threshold: float = 0.5
    device: str = 'cpu'

    # Bounding box computation
    scaling_factor: float = 1.5
    preserve_aspect_ratio: bool = True
    target_aspect_ratio: Optional[float] = None

    # Trajectory smoothing
    cutoff_hz: float = 2.0
    sample_rate: Optional[float] = None

    # Processing options
    frame_skip: int = 1
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid
        FileNotFoundError
            If input_video does not exist

        Examples
        --------
        >>> config = PostProcessingConfig('video.mp4', 'out.json')
        >>> config.validate()  # Raises if invalid
        """
        # Validate paths
        input_path = Path(self.input_video)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video}")

        output_path = Path(self.output_json)
        if not output_path.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_path.parent}"
            )

        # Validate output video path if specified
        if self.output_video:
            output_video_path = Path(self.output_video)
            if not output_video_path.parent.exists():
                raise FileNotFoundError(
                    f"Output video directory does not exist: {output_video_path.parent}"
                )

        # Validate numeric ranges
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be 0.0-1.0, got {self.confidence_threshold}"
            )

        if self.scaling_factor <= 0:
            raise ValueError(
                f"scaling_factor must be positive, got {self.scaling_factor}"
            )

        if self.cutoff_hz < 0:
            raise ValueError(
                f"cutoff_hz must be non-negative, got {self.cutoff_hz}"
            )

        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )

        if self.frame_skip < 1:
            raise ValueError(
                f"frame_skip must be >= 1, got {self.frame_skip}"
            )

        if self.target_aspect_ratio is not None and self.target_aspect_ratio <= 0:
            raise ValueError(
                f"target_aspect_ratio must be positive, got {self.target_aspect_ratio}"
            )

        # Validate device
        valid_devices = ['cpu', 'cuda', 'mps']
        if not any(self.device.startswith(d) for d in valid_devices):
            raise ValueError(
                f"device must be one of {valid_devices}, got {self.device}"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON serialization

        Examples
        --------
        >>> config = PostProcessingConfig('video.mp4', 'out.json')
        >>> config_dict = config.to_dict()
        >>> import json
        >>> json.dumps(config_dict)
        """
        return {
            'input_video': str(self.input_video),
            'output_json': str(self.output_json),
            'output_video': str(self.output_video) if self.output_video else None,
            'yolo_model': str(self.yolo_model),
            'confidence_threshold': self.confidence_threshold,
            'scaling_factor': self.scaling_factor,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'target_aspect_ratio': self.target_aspect_ratio,
            'cutoff_hz': self.cutoff_hz,
            'sample_rate': self.sample_rate,
            'frame_skip': self.frame_skip,
            'device': self.device,
            'verbose': self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PostProcessingConfig':
        """Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with configuration parameters

        Returns
        -------
        PostProcessingConfig
            Configuration instance

        Examples
        --------
        >>> config_dict = {
        ...     'input_video': 'clip.mp4',
        ...     'output_json': 'metadata.json',
        ...     'scaling_factor': 1.8
        ... }
        >>> config = PostProcessingConfig.from_dict(config_dict)
        """
        return cls(**config_dict)


__all__ = ['PostProcessingConfig']
