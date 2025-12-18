"""Cable path geometry and interpolation."""

import numpy as np
from typing import List, Tuple


class CablePath:
    """Represents a closed cable path through tower points.

    The path is a closed polyline that allows querying position and direction
    at any distance along the cable.
    """

    def __init__(self, tower_points: List[Tuple[float, float]]):
        """Initialize cable path from tower coordinates.

        Args:
            tower_points: List of (x, y) tower coordinates in meters
        """
        # Convert to numpy array for easier math
        self.towers = np.array(tower_points, dtype=np.float64)
        self.n_towers = len(self.towers)

        # Calculate segment vectors and lengths
        # Close the loop by connecting last tower back to first
        self.segments = []
        self.segment_lengths = []
        self.cumulative_lengths = [0.0]

        for i in range(self.n_towers):
            start = self.towers[i]
            end = self.towers[(i + 1) % self.n_towers]  # wrap around

            segment_vec = end - start
            segment_len = np.linalg.norm(segment_vec)

            self.segments.append(segment_vec)
            self.segment_lengths.append(segment_len)
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + segment_len
            )

        self.total_length = self.cumulative_lengths[-1]

        # Remove the last cumulative length (it's just total_length)
        self.cumulative_lengths = self.cumulative_lengths[:-1]

    def get_position_at_distance(self, s: float) -> Tuple[np.ndarray, int]:
        """Get position and segment index at distance s along the path.

        Args:
            s: Distance along the path in meters (wraps around)

        Returns:
            Tuple of (position as [x, y], segment_index)
        """
        # Wrap s to [0, total_length)
        s = s % self.total_length

        # Find which segment we're on
        segment_idx = self.n_towers - 1  # default to last segment
        for i in range(self.n_towers):
            if i < self.n_towers - 1:
                # For all segments except last, check if s < next cumulative length
                if s < self.cumulative_lengths[i + 1]:
                    segment_idx = i
                    break
            else:
                # Last segment: if we're here, we're on the last segment
                segment_idx = i

        # Calculate position within segment
        s_in_segment = s - self.cumulative_lengths[segment_idx]
        t = s_in_segment / self.segment_lengths[segment_idx]  # [0, 1]

        start_pos = self.towers[segment_idx]
        segment_vec = self.segments[segment_idx]

        position = start_pos + t * segment_vec

        return position, segment_idx

    def get_tangent_at_distance(self, s: float) -> np.ndarray:
        """Get unit tangent vector at distance s along the path.

        Args:
            s: Distance along the path in meters (wraps around)

        Returns:
            Unit tangent vector as [x, y]
        """
        # Wrap s to [0, total_length)
        s = s % self.total_length

        # Find which segment we're on
        segment_idx = self.n_towers - 1  # default to last segment
        for i in range(self.n_towers):
            if i < self.n_towers - 1:
                if s < self.cumulative_lengths[i + 1]:
                    segment_idx = i
                    break
            else:
                segment_idx = i

        # Get segment direction
        segment_vec = self.segments[segment_idx]
        tangent = segment_vec / np.linalg.norm(segment_vec)

        return tangent

    def get_segment_index_at_distance(self, s: float) -> int:
        """Get the segment index at distance s along the path.

        Args:
            s: Distance along the path in meters (wraps around)

        Returns:
            Segment index (0 to n_towers-1)
        """
        s = s % self.total_length

        for i in range(self.n_towers):
            if i < self.n_towers - 1:
                if s < self.cumulative_lengths[i + 1]:
                    return i
            else:
                return i

        return 0  # fallback (should not reach here)
