"""Gap detection for trajectory segmentation in annotation pipeline."""

import logging
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)


class GapEvent(NamedTuple):
    """Event returned by gap detection processing."""
    segment_started: bool  # True if new segment started
    segment_ended: bool    # True if current segment ended due to gap
    trajectory_id: int     # Current trajectory identifier
    is_first_detection: bool  # True if this is the very first detection


class GapDetector:
    """Real-time gap detection for trajectory segmentation.
    
    Purpose
    -------
    Manages gap detection state and logic for identifying when rider trajectories
    should be split into separate segments. Extracts complex gap timing logic
    from the main processing loop for better maintainability and testability.
    
    The detector tracks time since the last person detection and triggers new
    trajectory segments when gaps exceed the configured timeout threshold.
    """
    
    def __init__(self, gap_timeout_s: float, original_fps: float):
        """Initialize gap detection with timeout and frame rate parameters.
        
        Purpose
        -------
        Configure gap detection thresholds and initialize state tracking
        variables for real-time processing.
        
        Inputs
        ------
        gap_timeout_s: float
            Maximum time gap in seconds before starting a new trajectory segment.
        original_fps: float
            Original video frame rate used to convert timeout to frame count.
            
        Outputs
        -------
        None
            Initializes internal state for gap tracking.
        """
        # Configuration
        self.gap_frames = int(gap_timeout_s * original_fps)
        self.gap_timeout_s = gap_timeout_s
        
        # State tracking
        self.frames_since_last_detection = 0
        self.is_first_detection = True
        self.last_detection_frame: Optional[int] = None
        self.first_bbox_after_gap_frame: Optional[int] = None
        self.trajectory_id = 0
        
        logger.debug("Initialized GapDetector: timeout=%.1fs (%d frames at %.1f fps)", 
                    gap_timeout_s, self.gap_frames, original_fps)
    
    def process_frame(self, frame_idx: int, has_detection: bool, sample_rate: float) -> GapEvent:
        """Process a single frame and return gap detection events.
        
        Purpose
        -------
        Core gap detection logic that processes each frame to determine if
        trajectory segments should start or end based on detection presence
        and timing gaps.
        
        Inputs
        ------
        frame_idx: int
            Current video frame index being processed.
        has_detection: bool
            True if person detection found in current frame.
        sample_rate: float
            Frame sampling rate for accurate gap timing calculations.
            
        Outputs
        -------
        GapEvent
            Named tuple indicating segment state changes and trajectory ID.
        """
        segment_started = False
        segment_ended = False
        was_first = self.is_first_detection
        
        if has_detection:
            logger.debug("Detection at frame %d: %d person(s) found", frame_idx, 1)
            
            # Check if we're starting a new segment after a gap OR this is first detection
            gap_detected = (self.frames_since_last_detection >= self.gap_frames 
                          and self.last_detection_frame is not None)
            
            if gap_detected or self.is_first_detection:
                if not self.is_first_detection:
                    # Starting new segment after gap - previous segment will be generated
                    segment_started = True
                    logger.info("Starting new trajectory after gap at frame %d", frame_idx)
                else:
                    logger.info("First detection at frame %d - starting initial trajectory", frame_idx)
                    segment_started = True
                
                # Reset gap tracking for new segment
                self.first_bbox_after_gap_frame = None
                self.is_first_detection = False
                
            # Update detection state
            self.frames_since_last_detection = 0
            if self.first_bbox_after_gap_frame is None:
                self.first_bbox_after_gap_frame = frame_idx
            
            self.last_detection_frame = frame_idx
            
        else:
            # No detection - increment gap counter
            self.frames_since_last_detection += sample_rate
            logger.debug("No detection at frame %d, gap duration: %.1f frames (threshold: %d)", 
                        frame_idx, self.frames_since_last_detection, self.gap_frames)
            
            # Check if gap threshold just reached (segment just ended)
            if (self.frames_since_last_detection >= self.gap_frames 
                and self.last_detection_frame is not None):
                
                logger.info("Gap threshold reached at frame %d (%.1fs gap). Ending current trajectory", 
                           frame_idx, self.gap_timeout_s)
                segment_ended = True
        
        return GapEvent(
            segment_started=segment_started,
            segment_ended=segment_ended, 
            trajectory_id=self.trajectory_id,
            is_first_detection=was_first
        )
    
    def should_start_new_segment(self, has_detection: bool) -> bool:
        """Check if a new trajectory segment should start.
        
        Purpose
        -------
        Determine if current frame conditions warrant starting a new
        trajectory segment based on gap detection state.
        
        Inputs  
        ------
        has_detection: bool
            Whether person detection exists in current frame.
            
        Outputs
        -------
        bool
            True if new segment should start, False otherwise.
        """
        if not has_detection:
            return False
            
        gap_detected = (self.frames_since_last_detection >= self.gap_frames 
                       and self.last_detection_frame is not None)
        
        return gap_detected or self.is_first_detection
    
    def should_end_current_segment(self, has_detection: bool) -> bool:
        """Check if current trajectory segment should end due to gap.
        
        Purpose
        -------
        Determine if gap threshold has been exceeded and current
        trajectory segment should be finalized.
        
        Inputs
        ------
        has_detection: bool
            Whether person detection exists in current frame.
            
        Outputs
        -------
        bool
            True if current segment should end, False otherwise.
        """
        if has_detection:
            return False
            
        return (self.frames_since_last_detection >= self.gap_frames 
                and self.last_detection_frame is not None)
    
    def get_current_trajectory_id(self) -> int:
        """Get the current trajectory identifier.
        
        Purpose
        -------
        Provide access to current trajectory ID for labeling purposes.
        
        Outputs
        -------
        int
            Current trajectory identifier.
        """
        return self.trajectory_id + 1 if not self.is_first_detection else 1
    
    def get_last_detection_frame(self) -> Optional[int]:
        """Get frame index of most recent detection.
        
        Purpose
        -------
        Provide access to last detection frame for segment boundary trimming.
        
        Outputs
        -------
        int | None
            Frame index of last detection, or None if no detections yet.
        """
        return self.last_detection_frame
    
    def reset_for_new_segment(self) -> None:
        """Reset state for starting a new trajectory segment.
        
        Purpose
        -------
        Clear segment-specific state while maintaining trajectory counter
        and overall detection state.
        
        Outputs
        -------
        None
            Updates internal state variables.
        """
        self.first_bbox_after_gap_frame = None
        # Note: Don't reset frames_since_last_detection, is_first_detection, 
        # or trajectory_id as these track global state