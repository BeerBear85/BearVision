"""Shared trajectory processing library for wakeboard video analysis.

This module provides a stable API for trajectory smoothing operations used across
the BearVision system, including the annotation pipeline and post-processing pipeline.

Purpose
-------
Refactored from the YOLO trainer to provide reusable trajectory interpolation
and smoothing functionality with a clean, well-documented interface.

Classes
-------
Detection : Dataclass representing a single YOLO detection
TrajectoryPoint : Dataclass representing a smoothed trajectory point
TrajectoryProcessor : Main API for trajectory interpolation and smoothing

Usage Example
-------------
>>> from TrajectoryProcessor import TrajectoryProcessor, Detection
>>> detections = [
...     Detection(frame_idx=0, cx=100, cy=200, w=50, h=100, confidence=0.9),
...     Detection(frame_idx=10, cx=120, cy=210, w=55, h=105, confidence=0.85),
... ]
>>> processor = TrajectoryProcessor(cutoff_hz=2.0, sample_rate=30.0)
>>> trajectory = processor.compute_smoothed_trajectory(detections, frame_range=(0, 10))
>>> for point in trajectory:
...     print(f"Frame {point.frame_idx}: ({point.x}, {point.y})")
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt


@dataclass
class Detection:
    """Represents a single YOLO detection in a video frame.

    Attributes
    ----------
    frame_idx : int
        Frame index in the video (0-based)
    cx : float
        Center x-coordinate of bounding box
    cy : float
        Center y-coordinate of bounding box
    w : float
        Width of bounding box
    h : float
        Height of bounding box
    confidence : float
        Detection confidence score (0.0 to 1.0)
    class_id : int, optional
        Class ID from YOLO model (default: 0 for 'person')
    label : str, optional
        Class label (default: 'person')
    """
    frame_idx: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    class_id: int = 0
    label: str = 'person'

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert center/size format to x1,y1,x2,y2 bounding box."""
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        x2 = self.cx + self.w / 2
        y2 = self.cy + self.h / 2
        return (x1, y1, x2, y2)


@dataclass
class TrajectoryPoint:
    """Represents a single point in a smoothed trajectory.

    Attributes
    ----------
    frame_idx : int
        Frame index in the video (0-based)
    x : float
        Smoothed x-coordinate (center of person)
    y : float
        Smoothed y-coordinate (center of person)
    w : float, optional
        Interpolated width of bounding box
    h : float, optional
        Interpolated height of bounding box
    """
    frame_idx: int
    x: float
    y: float
    w: Optional[float] = None
    h: Optional[float] = None


class TrajectoryProcessor:
    """Main API for trajectory interpolation and smoothing.

    This class provides methods to compute smoothed trajectories from sparse
    YOLO detections using cubic spline interpolation and low-pass filtering.

    Purpose
    -------
    Handles the mathematical operations required to convert discrete detection
    points into smooth, continuous trajectories suitable for video cropping
    and tracking visualization.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency for low-pass filter in Hertz. Higher values allow
        more high-frequency motion; lower values produce smoother trajectories.
        Set to 0.0 to disable filtering. Typical range: 0.5 to 5.0 Hz.
    sample_rate : float
        Sampling rate in frames per second. Should match the effective FPS
        of the detection stream (may differ from video FPS if frames are skipped).
    filter_order : int, optional
        Order of the Butterworth filter (default: 1). Higher orders provide
        steeper rolloff but may introduce artifacts.

    Attributes
    ----------
    cutoff_hz : float
        Low-pass filter cutoff frequency
    sample_rate : float
        Sampling rate in frames per second
    filter_order : int
        Butterworth filter order

    Methods
    -------
    compute_smoothed_trajectory(detections, frame_range)
        Main API: Compute smoothed trajectory from detections
    interpolate_positions(detections, frame_indices)
        Interpolate x,y positions using cubic splines
    interpolate_sizes(detections, frame_indices)
        Interpolate w,h sizes using cubic splines
    apply_lowpass_filter(sequence)
        Apply zero-phase low-pass filtering to a sequence

    Usage
    -----
    >>> processor = TrajectoryProcessor(cutoff_hz=2.0, sample_rate=30.0)
    >>> trajectory = processor.compute_smoothed_trajectory(detections, (0, 100))
    """

    def __init__(self, cutoff_hz: float = 2.0, sample_rate: float = 30.0,
                 filter_order: int = 1):
        """Initialize the trajectory processor.

        Parameters
        ----------
        cutoff_hz : float, default 2.0
            Cutoff frequency for low-pass filter in Hertz
        sample_rate : float, default 30.0
            Sampling rate in frames per second
        filter_order : int, default 1
            Order of the Butterworth filter
        """
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        self.filter_order = filter_order

    def apply_lowpass_filter(self, sequence: List[float]) -> List[float]:
        """Apply zero-phase low-pass filtering to a sequence.

        Purpose
        -------
        Suppress high-frequency jitter in the trajectory without introducing
        phase delay, keeping filtered positions aligned with their original frames.
        Uses a Butterworth filter applied forward and backward (filtfilt) for
        zero-phase response.

        Parameters
        ----------
        sequence : list of float
            Sequence of samples representing either x or y coordinates

        Returns
        -------
        list of float
            Filtered sequence retaining the original length

        Notes
        -----
        - Returns unfiltered sequence if cutoff_hz <= 0 (filtering disabled)
        - Returns unfiltered sequence if length < 13 (too short for filtfilt padding)
        - Uses scipy.signal.filtfilt for zero-phase filtering
        """
        if self.cutoff_hz <= 0 or len(sequence) == 0:
            return sequence

        # filtfilt requires minimum length for padding
        if len(sequence) < 13:
            return sequence

        nyq = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_hz / nyq
        b, a = butter(self.filter_order, normal_cutoff, btype="low", analog=False)

        # filtfilt runs the filter forward and backward, eliminating phase delay
        return filtfilt(b, a, sequence).tolist()

    def interpolate_positions(self, detections: List[Detection],
                             frame_indices: List[int]) -> Tuple[List[float], List[float]]:
        """Interpolate x,y positions using cubic spline interpolation.

        Purpose
        -------
        Fill in missing trajectory points between sparse detections using
        smooth cubic splines. This creates natural-looking motion paths even
        when detections are not available for every frame.

        Parameters
        ----------
        detections : list of Detection
            Sparse detection points to interpolate between
        frame_indices : list of int
            Frame indices where interpolated positions are needed

        Returns
        -------
        tuple of (list of float, list of float)
            Interpolated (x_positions, y_positions) for requested frame indices

        Raises
        ------
        ValueError
            If fewer than 2 detections provided (need at least 2 for spline)

        Notes
        -----
        - Uses scipy.interpolate.CubicSpline for smooth interpolation
        - Extrapolation beyond detection range uses edge values (no wild extrapolation)
        """
        if len(detections) < 2:
            raise ValueError("Need at least 2 detections for interpolation")

        det_frames = [d.frame_idx for d in detections]
        det_xs = [d.cx for d in detections]
        det_ys = [d.cy for d in detections]

        spline_x = CubicSpline(det_frames, det_xs)
        spline_y = CubicSpline(det_frames, det_ys)

        interp_xs = [float(spline_x(fi)) for fi in frame_indices]
        interp_ys = [float(spline_y(fi)) for fi in frame_indices]

        return interp_xs, interp_ys

    def interpolate_sizes(self, detections: List[Detection],
                         frame_indices: List[int]) -> Tuple[List[float], List[float]]:
        """Interpolate bounding box sizes using cubic spline interpolation.

        Purpose
        -------
        Smoothly interpolate bounding box width and height between detections
        to avoid abrupt size changes in tracked objects.

        Parameters
        ----------
        detections : list of Detection
            Sparse detection points to interpolate between
        frame_indices : list of int
            Frame indices where interpolated sizes are needed

        Returns
        -------
        tuple of (list of float, list of float)
            Interpolated (widths, heights) for requested frame indices

        Raises
        ------
        ValueError
            If fewer than 2 detections provided

        Notes
        -----
        Uses cubic spline interpolation for smooth size transitions
        """
        if len(detections) < 2:
            raise ValueError("Need at least 2 detections for interpolation")

        det_frames = [d.frame_idx for d in detections]
        det_ws = [d.w for d in detections]
        det_hs = [d.h for d in detections]

        spline_w = CubicSpline(det_frames, det_ws)
        spline_h = CubicSpline(det_frames, det_hs)

        interp_ws = [float(spline_w(fi)) for fi in frame_indices]
        interp_hs = [float(spline_h(fi)) for fi in frame_indices]

        return interp_ws, interp_hs

    def compute_smoothed_trajectory(
        self,
        detections: List[Detection],
        frame_range: Optional[Tuple[int, int]] = None,
        include_sizes: bool = True
    ) -> List[TrajectoryPoint]:
        """Compute smoothed trajectory from sparse detections.

        This is the main public API for trajectory processing. It combines
        cubic spline interpolation with low-pass filtering to produce a
        smooth, continuous trajectory from sparse YOLO detections.

        Purpose
        -------
        Convert discrete detection points into a smooth trajectory suitable
        for video cropping, tracking visualization, or motion analysis.

        Parameters
        ----------
        detections : list of Detection
            Sparse detection points (must have at least 2 detections)
        frame_range : tuple of (int, int), optional
            (start_frame, end_frame) inclusive range for trajectory generation.
            If None, uses range from first to last detection.
        include_sizes : bool, default True
            Whether to interpolate bounding box sizes (w, h) in addition to positions

        Returns
        -------
        list of TrajectoryPoint
            Smoothed trajectory points with frame index and coordinates

        Raises
        ------
        ValueError
            If fewer than 2 detections provided

        Algorithm
        ---------
        1. Determine frame range from detections or provided range
        2. Interpolate x,y positions using cubic splines
        3. Optionally interpolate w,h sizes using cubic splines
        4. Apply low-pass filter to x and y sequences
        5. Combine into TrajectoryPoint objects

        Examples
        --------
        >>> detections = [
        ...     Detection(0, 100.0, 200.0, 50.0, 100.0, 0.9),
        ...     Detection(10, 120.0, 210.0, 52.0, 102.0, 0.85),
        ... ]
        >>> processor = TrajectoryProcessor(cutoff_hz=2.0, sample_rate=30.0)
        >>> trajectory = processor.compute_smoothed_trajectory(detections)
        >>> print(f"Generated {len(trajectory)} trajectory points")
        """
        if len(detections) < 2:
            # Single detection case: return trajectory with just that point
            if len(detections) == 1:
                d = detections[0]
                return [TrajectoryPoint(d.frame_idx, d.cx, d.cy, d.w, d.h)]
            raise ValueError("Need at least 1 detection for trajectory")

        # Determine frame range
        if frame_range is None:
            first_frame = min(d.frame_idx for d in detections)
            last_frame = max(d.frame_idx for d in detections)
        else:
            first_frame, last_frame = frame_range

        frame_indices = list(range(first_frame, last_frame + 1))

        # Interpolate positions
        interp_xs, interp_ys = self.interpolate_positions(detections, frame_indices)

        # Interpolate sizes if requested
        if include_sizes:
            interp_ws, interp_hs = self.interpolate_sizes(detections, frame_indices)
        else:
            interp_ws = [None] * len(frame_indices)
            interp_hs = [None] * len(frame_indices)

        # Apply low-pass filter to positions
        smoothed_xs = self.apply_lowpass_filter(interp_xs)
        smoothed_ys = self.apply_lowpass_filter(interp_ys)

        # Create trajectory points
        trajectory = []
        for fi, x, y, w, h in zip(frame_indices, smoothed_xs, smoothed_ys,
                                   interp_ws, interp_hs):
            trajectory.append(TrajectoryPoint(fi, x, y, w, h))

        return trajectory


# Legacy compatibility functions for existing code
def lowpass_filter(seq: List[float], cutoff_hz: float, sample_rate: float) -> List[float]:
    """Legacy compatibility wrapper for low-pass filtering.

    This function maintains backward compatibility with existing code that
    directly calls lowpass_filter(). New code should use TrajectoryProcessor.

    Parameters
    ----------
    seq : list of float
        Sequence to filter
    cutoff_hz : float
        Cutoff frequency in Hz
    sample_rate : float
        Sampling rate in Hz

    Returns
    -------
    list of float
        Filtered sequence
    """
    processor = TrajectoryProcessor(cutoff_hz=cutoff_hz, sample_rate=sample_rate)
    return processor.apply_lowpass_filter(seq)


__all__ = [
    'Detection',
    'TrajectoryPoint',
    'TrajectoryProcessor',
    'lowpass_filter',  # Legacy compatibility
]
