"""Bounding box processing for virtual cameraman functionality.

This module handles fixed-size bounding box computation and per-frame box
generation with frame boundary clamping for the post-processing pipeline.

Purpose
-------
Implements the "virtual cameraman" logic that determines optimal fixed-size
bounding boxes for cropping wakeboard videos, ensuring the rider stays centered
and fully visible throughout the clip.

Classes
-------
BoundingBox : Dataclass representing a rectangular bounding box
FixedBoxSize : Dataclass representing fixed box dimensions
BoundingBoxProcessor : Main API for box computation and generation

Usage Example
-------------
>>> from BoundingBoxProcessor import BoundingBoxProcessor
>>> from TrajectoryProcessor import TrajectoryPoint
>>>
>>> trajectory = [
...     TrajectoryPoint(0, 960, 540, 100, 200),
...     TrajectoryPoint(1, 965, 545, 105, 205),
... ]
>>>
>>> processor = BoundingBoxProcessor(
...     frame_width=1920,
...     frame_height=1080,
...     scaling_factor=1.5
... )
>>>
>>> fixed_size = processor.compute_fixed_box_size(trajectory)
>>> per_frame_boxes = processor.generate_per_frame_boxes(trajectory, fixed_size)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class BoundingBox:
    """Represents a rectangular bounding box.

    Attributes
    ----------
    x1 : float
        Left edge x-coordinate
    y1 : float
        Top edge y-coordinate
    x2 : float
        Right edge x-coordinate
    y2 : float
        Bottom edge y-coordinate

    Methods
    -------
    width : float
        Box width (x2 - x1)
    height : float
        Box height (y2 - y1)
    center : tuple of (float, float)
        Box center coordinates (cx, cy)
    area : float
        Box area (width * height)
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Calculate box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculate box height."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate box center coordinates."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (cx, cy)

    @property
    def area(self) -> float:
        """Calculate box area."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'x1': float(self.x1),
            'y1': float(self.y1),
            'x2': float(self.x2),
            'y2': float(self.y2),
            'width': float(self.width),
            'height': float(self.height),
            'center_x': float(self.center[0]),
            'center_y': float(self.center[1]),
        }


@dataclass
class FixedBoxSize:
    """Represents fixed bounding box dimensions.

    Attributes
    ----------
    width : float
        Fixed box width in pixels
    height : float
        Fixed box height in pixels
    aspect_ratio : float
        Width / height ratio
    """

    width: float
    height: float

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'width': float(self.width),
            'height': float(self.height),
            'aspect_ratio': float(self.aspect_ratio),
        }


class BoundingBoxProcessor:
    """Main API for fixed-size bounding box computation and generation.

    This class implements the virtual cameraman logic for determining optimal
    crop boxes that keep the rider centered while maintaining consistent framing.

    Purpose
    -------
    Given a trajectory of rider positions, compute a fixed-size bounding box
    that encompasses all observed positions with appropriate padding, then
    generate per-frame boxes centered on the trajectory while respecting
    frame boundaries.

    Parameters
    ----------
    frame_width : int
        Video frame width in pixels
    frame_height : int
        Video frame height in pixels
    scaling_factor : float, default 1.5
        Multiplicative factor applied to max observed box size.
        Values > 1.0 add padding around the rider.
    preserve_aspect_ratio : bool, default True
        Whether to preserve aspect ratio when scaling
    target_aspect_ratio : float, optional
        Desired aspect ratio (width/height) for output boxes.
        If None, uses aspect ratio of scaled max box.

    Attributes
    ----------
    frame_width : int
        Video frame width
    frame_height : int
        Video frame height
    scaling_factor : float
        Box size scaling factor
    preserve_aspect_ratio : bool
        Aspect ratio preservation flag
    target_aspect_ratio : float or None
        Target aspect ratio for output boxes

    Methods
    -------
    compute_fixed_box_size(trajectory)
        Determine fixed box size from trajectory
    generate_per_frame_boxes(trajectory, fixed_size)
        Generate frame-by-frame boxes centered on trajectory
    clamp_box_to_frame(box)
        Ensure box stays within frame boundaries

    Usage
    -----
    >>> processor = BoundingBoxProcessor(1920, 1080, scaling_factor=1.5)
    >>> fixed_size = processor.compute_fixed_box_size(trajectory)
    >>> boxes = processor.generate_per_frame_boxes(trajectory, fixed_size)
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        scaling_factor: float = 1.5,
        preserve_aspect_ratio: bool = True,
        target_aspect_ratio: Optional[float] = None
    ):
        """Initialize the bounding box processor.

        Parameters
        ----------
        frame_width : int
            Video frame width in pixels
        frame_height : int
            Video frame height in pixels
        scaling_factor : float, default 1.5
            Multiplicative factor for box size scaling
        preserve_aspect_ratio : bool, default True
            Whether to preserve aspect ratio when scaling
        target_aspect_ratio : float, optional
            Desired aspect ratio (width/height) for output boxes
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scaling_factor = scaling_factor
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.target_aspect_ratio = target_aspect_ratio

    def compute_fixed_box_size(self, trajectory: List) -> FixedBoxSize:
        """Compute fixed bounding box size from trajectory.

        This method determines the optimal fixed box size by:
        1. Finding the maximum observed bounding box dimensions across all frames
        2. Scaling by the configured scaling_factor (default 1.5)
        3. Optionally adjusting to match target aspect ratio

        Purpose
        -------
        Create a consistent crop size that can accommodate the rider's motion
        throughout the entire clip with appropriate padding.

        Parameters
        ----------
        trajectory : list of TrajectoryPoint
            Smoothed trajectory with interpolated positions and sizes.
            Must have w and h attributes (from TrajectoryProcessor with include_sizes=True)

        Returns
        -------
        FixedBoxSize
            Fixed dimensions for all frames in the clip

        Raises
        ------
        ValueError
            If trajectory is empty or trajectory points lack size information

        Algorithm
        ---------
        1. Find max width and max height across all trajectory points
        2. Take the maximum of (max_width, max_height) for square-ish box
        3. Scale by scaling_factor
        4. If target_aspect_ratio specified, adjust dimensions to match
        5. Clamp to frame dimensions if necessary

        Notes
        -----
        - Using max(width, height) creates a more square box that accommodates
          rider rotation and trick variations
        - Scaling factor typically 1.2-2.0 depending on desired framing
        - Result may be clamped to frame dimensions for very large riders/scaling

        Examples
        --------
        >>> from TrajectoryProcessor import TrajectoryPoint
        >>> trajectory = [
        ...     TrajectoryPoint(0, 960, 540, 100, 200),
        ...     TrajectoryPoint(1, 965, 545, 120, 220),
        ... ]
        >>> processor = BoundingBoxProcessor(1920, 1080, scaling_factor=1.5)
        >>> fixed_size = processor.compute_fixed_box_size(trajectory)
        >>> print(f"Fixed box: {fixed_size.width}x{fixed_size.height}")
        """
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        # Extract maximum observed dimensions
        max_width = 0.0
        max_height = 0.0

        for point in trajectory:
            if point.w is None or point.h is None:
                raise ValueError(
                    "Trajectory points must include size information (w, h). "
                    "Use TrajectoryProcessor with include_sizes=True."
                )
            max_width = max(max_width, point.w)
            max_height = max(max_height, point.h)

        # Take the maximum dimension for a more square box
        # This accommodates rider rotation and orientation changes
        max_dimension = max(max_width, max_height)

        # Scale by configured factor
        scaled_dimension = max_dimension * self.scaling_factor

        # Determine final box dimensions
        if self.target_aspect_ratio is not None:
            # Use target aspect ratio
            if self.target_aspect_ratio >= 1.0:
                # Landscape or square
                box_width = scaled_dimension
                box_height = box_width / self.target_aspect_ratio
            else:
                # Portrait
                box_height = scaled_dimension
                box_width = box_height * self.target_aspect_ratio
        elif self.preserve_aspect_ratio:
            # Preserve detected aspect ratio
            detected_aspect = max_width / max_height if max_height > 0 else 1.0
            if detected_aspect >= 1.0:
                box_width = scaled_dimension
                box_height = box_width / detected_aspect
            else:
                box_height = scaled_dimension
                box_width = box_height * detected_aspect
        else:
            # Square box (aspect ratio 1:1)
            box_width = scaled_dimension
            box_height = scaled_dimension

        # Clamp to frame dimensions
        box_width = min(box_width, self.frame_width)
        box_height = min(box_height, self.frame_height)

        return FixedBoxSize(width=box_width, height=box_height)

    def clamp_box_to_frame(self, box: BoundingBox) -> BoundingBox:
        """Ensure bounding box stays within frame boundaries.

        If the box would extend beyond frame edges, shift it inward while
        preserving the box size. This prevents cropping operations from
        accessing pixels outside the valid frame area.

        Purpose
        -------
        Handle edge cases where the trajectory approaches frame boundaries,
        ensuring all generated boxes are valid and fully contained within
        the source frame.

        Parameters
        ----------
        box : BoundingBox
            Bounding box that may exceed frame boundaries

        Returns
        -------
        BoundingBox
            Clamped box fully contained within frame boundaries

        Algorithm
        ---------
        1. Check if box extends beyond left/top edges, shift right/down if so
        2. Check if box extends beyond right/bottom edges, shift left/up if so
        3. If box is larger than frame (shouldn't happen), clamp to frame size

        Notes
        -----
        - Preserves box size when possible
        - Shifts box position to keep it within frame
        - Only changes size if box is larger than frame (fallback case)

        Examples
        --------
        >>> processor = BoundingBoxProcessor(1920, 1080)
        >>> box = BoundingBox(-10, 50, 190, 250)  # Extends beyond left edge
        >>> clamped = processor.clamp_box_to_frame(box)
        >>> print(f"Clamped box: ({clamped.x1}, {clamped.y1}) to ({clamped.x2}, {clamped.y2})")
        """
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        box_width = box.width
        box_height = box.height

        # Clamp to left/top edges
        if x1 < 0:
            x1 = 0
            x2 = box_width
        if y1 < 0:
            y1 = 0
            y2 = box_height

        # Clamp to right/bottom edges
        if x2 > self.frame_width:
            x2 = self.frame_width
            x1 = self.frame_width - box_width
        if y2 > self.frame_height:
            y2 = self.frame_height
            y1 = self.frame_height - box_height

        # Final clamp if box is larger than frame (shouldn't happen normally)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.frame_width, x2)
        y2 = min(self.frame_height, y2)

        return BoundingBox(x1, y1, x2, y2)

    def generate_per_frame_boxes(
        self,
        trajectory: List,
        fixed_size: FixedBoxSize
    ) -> List[Tuple[int, BoundingBox]]:
        """Generate per-frame bounding boxes centered on trajectory.

        For each point in the trajectory, create a fixed-size bounding box
        centered on the rider's position and clamp it to stay within frame
        boundaries.

        Purpose
        -------
        Create the final per-frame crop boxes that will be used for video
        rendering or metadata export. Each box maintains consistent size
        while following the smoothed trajectory.

        Parameters
        ----------
        trajectory : list of TrajectoryPoint
            Smoothed trajectory points with frame indices and positions
        fixed_size : FixedBoxSize
            Fixed dimensions to use for all boxes

        Returns
        -------
        list of (int, BoundingBox)
            List of (frame_idx, box) tuples for each trajectory point

        Algorithm
        ---------
        1. For each trajectory point:
           a. Center box on (x, y) position
           b. Apply fixed width and height
           c. Clamp box to frame boundaries
        2. Return list of frame_idx and box pairs

        Notes
        -----
        - All boxes have identical dimensions (from fixed_size)
        - Box position varies to track the trajectory
        - Clamping ensures boxes never extend beyond frame edges

        Examples
        --------
        >>> trajectory = [
        ...     TrajectoryPoint(0, 960, 540, 100, 200),
        ...     TrajectoryPoint(1, 965, 545, 100, 200),
        ... ]
        >>> fixed_size = FixedBoxSize(width=300, height=400)
        >>> processor = BoundingBoxProcessor(1920, 1080)
        >>> boxes = processor.generate_per_frame_boxes(trajectory, fixed_size)
        >>> for frame_idx, box in boxes:
        ...     print(f"Frame {frame_idx}: {box.x1}, {box.y1}, {box.x2}, {box.y2}")
        """
        per_frame_boxes = []

        half_width = fixed_size.width / 2
        half_height = fixed_size.height / 2

        for point in trajectory:
            # Center box on trajectory point
            x1 = point.x - half_width
            y1 = point.y - half_height
            x2 = point.x + half_width
            y2 = point.y + half_height

            # Create box and clamp to frame
            box = BoundingBox(x1, y1, x2, y2)
            clamped_box = self.clamp_box_to_frame(box)

            per_frame_boxes.append((point.frame_idx, clamped_box))

        return per_frame_boxes


__all__ = [
    'BoundingBox',
    'FixedBoxSize',
    'BoundingBoxProcessor',
]
