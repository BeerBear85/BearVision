"""Trajectory generation and filtering for annotation pipeline."""
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

# Import configuration classes
from annotation_config import PipelineConfig, TrajectoryConfig


logger = logging.getLogger(__name__)


def lowpass_filter(seq: List[float], cutoff_hz: float, sample_rate: float) -> List[float]:
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

    if cutoff_hz <= 0 or len(seq) == 0:
        # Users can disable filtering by setting the cutoff to zero.
        # Also return original sequence if it's empty to avoid filtfilt errors.
        return seq
    
    # filtfilt requires minimum length for padding, return unfiltered if too short
    if len(seq) < 13:  # filtfilt typically needs at least 3*padlen where padlen is 4 for order 1
        return seq
        
    order = 1  # First order keeps attenuation mild and preserves realism.
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # ``filtfilt`` runs the filter forward and backward, eliminating phase
    # delay so that annotations remain time-aligned.
    return filtfilt(b, a, seq).tolist()


def save_trajectory_image(
    trajectory: List[Tuple[int, int]],
    segment_items: List[Dict[str, Any]],
    output_dir: str,
    track_id: int,
) -> str | None:
    """Save a trajectory visualization image.
    
    Purpose
    -------
    Generate and save a trajectory image when a segment is completed,
    allowing the GUI to display trajectory previews. Uses the middle frame
    of the trajectory segment with a red line overlay showing the rider's position.
    
    Inputs
    ------
    trajectory: List[Tuple[int, int]]
        List of (x, y) trajectory points.
    segment_items: List[Dict[str, Any]]
        All frame items in the trajectory segment, used to select the middle frame.
    output_dir: str
        Output directory where trajectories folder will be created.
    track_id: int
        Unique identifier for this trajectory.
        
    Outputs
    -------
    str | None
        Path to the saved trajectory image, or None if saving failed.
    """
    # Check if trajectory is empty - handle both lists and arrays
    if (not trajectory if isinstance(trajectory, list) else len(trajectory) == 0) or not segment_items:
        return None
    
    # Ensure trajectory is a proper list of tuples
    if not isinstance(trajectory, list):
        return None
        
    try:
        # Create trajectories directory
        trajectories_dir = Path(output_dir) / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        
        # Select the middle frame based on frame_idx, not list position
        first_frame_idx = int(segment_items[0]["frame_idx"])
        last_frame_idx = int(segment_items[-1]["frame_idx"])
        middle_frame_idx = (first_frame_idx + last_frame_idx) // 2
        
        # Find the segment item that best matches the middle frame_idx
        middle_item = min(segment_items, key=lambda item: abs(int(item["frame_idx"]) - middle_frame_idx))
        
        # Create trajectory visualization on the middle frame
        disp = middle_item["frame"].copy() # Copy to avoid altering the original frame
        
        # Ensure trajectory is a list of tuples and not arrays
        if isinstance(trajectory, np.ndarray):
            trajectory = trajectory.tolist()
        trajectory = [(int(x), int(y)) for x, y in trajectory]
        
        pts = np.array(trajectory, dtype=int) # Convert to integer for drawing
        cv2.polylines(disp, [pts], False, (0, 0, 255), 2) # Draw trajectory in red
        
        # Draw a red line on the middle frame showing the rider's position
        # Find the trajectory point that corresponds to the selected middle frame
        # Use frame_idx to find the index since numpy arrays make direct comparison fail
        middle_list_idx = None
        for i, item in enumerate(segment_items):
            if item["frame_idx"] == middle_item["frame_idx"]:
                middle_list_idx = i
                break
        
        if middle_list_idx is not None and middle_list_idx < len(trajectory):
            middle_point = trajectory[middle_list_idx]
            # Draw a cross-hair or line to mark the rider's position at this frame
            x, y = middle_point
            line_length = 20  # Length of the cross-hair lines
            # Draw horizontal line
            cv2.line(disp, (x - line_length, y), (x + line_length, y), (0, 0, 255), 3)
            # Draw vertical line
            cv2.line(disp, (x, y - line_length), (x, y + line_length), (0, 0, 255), 3)
        
        # Generate unique filename with timestamp to ensure GUI detects new files
        timestamp = int(time.time() * 1000)  # millisecond precision
        trajectory_path = trajectories_dir / f"trajectory_{track_id}_{timestamp}.jpg"
        
        # Save the trajectory image
        cv2.imwrite(str(trajectory_path), disp)
        return str(trajectory_path)
        
    except Exception as e:
        logger.warning(f"Failed to save trajectory image: {e}")
        return None


def generate_trajectory_during_processing(
    segment_items: List[Dict[str, Any]],
    det_points: List[Tuple[int, float, float, float, float, int, str]],
    cfg: PipelineConfig,
    track_id: int,
    sample_rate: float,
) -> Tuple[str | None, List[Tuple[int, int]], Dict[str, Any] | None]:
    """Generate and save trajectory image during processing when gap is detected.
    
    Purpose
    -------
    Create trajectory visualization immediately when detection gap exceeds threshold,
    enabling real-time trajectory generation during video processing instead of only
    at the end.
    
    Inputs
    ------
    segment_items: List[Dict[str, Any]]
        Frames belonging to the current trajectory segment.
    det_points: List[Tuple[int, float, float, float, float, int, str]]
        Detection tuples for the segment.
    cfg: PipelineConfig
        Full pipeline configuration.
    track_id: int
        Unique identifier for this trajectory.
    sample_rate: float
        Effective frames-per-second used for interpolation.
        
    Outputs
    -------
    Tuple[str | None, List[Tuple[int, int]], Dict[str, Any] | None]
        Tuple containing:
        - Path to the saved trajectory image, or None if generation failed
        - Computed trajectory points for reuse during export
        - Final item for preview, or None
    """
    if not det_points or not segment_items:
        return None, [], None
        
    try:
        # Interpolate trajectory points using existing spline logic
        trajectory: List[Tuple[int, int]] = []
        final_item = None
        
        if len(det_points) >= 2:
            frames = [p[0] for p in det_points]
            cxs = [p[1] for p in det_points]
            cys = [p[2] for p in det_points]
            sx = CubicSpline(frames, cxs)
            sy = CubicSpline(frames, cys)
            idx_map = {it["frame_idx"]: it for it in segment_items}
            first = segment_items[0]["frame_idx"]
            last = segment_items[-1]["frame_idx"]
            
            interp_points = []
            for fi in range(first, last + 1):
                item = idx_map.get(fi)
                if item is None:
                    continue
                cx = float(sx(fi))
                cy = float(sy(fi))
                interp_points.append((cx, cy, item))
            
            # Apply low-pass filtering
            xs = lowpass_filter([p[0] for p in interp_points], cfg.trajectory.cutoff_hz, sample_rate)
            ys = lowpass_filter([p[1] for p in interp_points], cfg.trajectory.cutoff_hz, sample_rate)
            
            for (cx, cy, item), fx, fy in zip(interp_points, xs, ys):
                trajectory.append((int(fx), int(fy)))
                final_item = item
        else:
            # Single detection point - just use the detection directly
            for item in segment_items:
                if item.get("boxes"):
                    b = item["boxes"][0]
                    x1, y1, x2, y2 = b["bbox"]
                    cx = int(x1 + (x2 - x1) / 2)
                    cy = int(y1 + (y2 - y1) / 2)
                    trajectory.append((cx, cy))
                    final_item = item
        
        # Generate and save trajectory image
        if trajectory and segment_items:
            image_path = save_trajectory_image(trajectory, segment_items, cfg.export.output_dir, track_id)
            return image_path, trajectory, final_item
        else:
            return None, trajectory, final_item
            
    except Exception as e:
        logger.warning(f"Failed to generate trajectory during processing: {e}")
    
    return None, [], None


def export_segment(
    segment_items: List[Dict[str, Any]],
    det_points: List[Tuple[int, float, float, float, float, int, str]],
    exporter: Any,  # DatasetExporter | CvatExporter
    cfg: PipelineConfig,
    sample_rate: float,
    track_id: int,
    frame_callback: Callable[[np.ndarray], None] | None = None,
    gui_mode: bool = False,
    pre_computed_trajectory: List[Tuple[int, int]] | None = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any] | None]:
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
    Tuple[List[Tuple[int, int]], Dict[str, Any] | None]
        The smoothed trajectory and the last frame processed, used for
        optional preview rendering.
    """

    trajectory: List[Tuple[int, int]] = []
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
        interp: List[Tuple[int, float, float, float, float, float, Dict[str, Any]]] = []
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

        # Use pre-computed trajectory if available, otherwise calculate it
        if pre_computed_trajectory and len(pre_computed_trajectory) == len(interp):
            # Reuse pre-computed filtered trajectory points
            filtered_coords = pre_computed_trajectory
        else:
            # Calculate trajectory if not pre-computed or length mismatch
            xs = lowpass_filter([p[1] for p in interp], cfg.trajectory.cutoff_hz, sample_rate)
            ys = lowpass_filter([p[2] for p in interp], cfg.trajectory.cutoff_hz, sample_rate)
            filtered_coords = list(zip(xs, ys))

        for (_fi, _cx, _cy, w, h, conf, item), (cx, cy) in zip(interp, filtered_coords):
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
            if frame_callback:
                # Show every interpolated frame even if seen before so users can
                # follow the smoothing process step by step.
                disp = item["frame"].copy()  # Copy so overlays don't alter saved data.
                if gui_mode:
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
                if frame_callback:
                    disp = item["frame"].copy()  # Avoid mutating frame that gets written.
                    if gui_mode:
                        for b in item.get("boxes", []):
                            x1, y1, x2, y2 = map(int, b["bbox"])
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame_callback(disp)

    # Save trajectory image when segment is completed and in GUI mode
    if trajectory and segment_items and gui_mode:
        save_trajectory_image(trajectory, segment_items, cfg.export.output_dir, track_id)

    return trajectory, final_item