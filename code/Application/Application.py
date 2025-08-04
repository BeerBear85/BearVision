"""Core application orchestration for the BearVision project."""

import logging, os
import MotionStartDetector, UserHandler, MotionTimeUserMatching, FullClipExtractor, TrackerClipExtractor
from Enums import ActionOptions, ClipTypes

logger = logging.getLogger(__name__)  # Set logger to reflect the current file


class Application:
    """High level controller coordinating BearVision processing steps."""

    def __init__(self) -> None:
        """Initialise the application instance."""
        logger.debug("Application created")

    def run(self, arg_input_video_folder: str, arg_user_root_folder: str, arg_selection: list[int]) -> None:
        """Execute the selected processing steps.

        Parameters
        ----------
        arg_input_video_folder : str
            Path to folder containing source videos.
        arg_user_root_folder : str
            Path to root directory holding user data.
        arg_selection : list[int]
            Indices of actions chosen in the GUI list box.

        Returns
        -------
        None
            The side effects are written files and extracted clips.
        """
        logger.info(
            "Running Application with video folder: %s user folder: %s\n",
            arg_input_video_folder,
            arg_user_root_folder,
        )

        # Create new helper objects for each run to avoid shared state between invocations.
        tmp_motion_start_detector = MotionStartDetector.MotionStartDetector()
        tmp_user_handler = UserHandler.UserHandler()
        tmp_motion_time_user_matching = MotionTimeUserMatching.MotionTimeUserMatching()
        tmp_full_clip_cut_extractor = FullClipExtractor.FullClipExtractor()
        tmp_tracker_clip_extractor = TrackerClipExtractor.TrackerClipExtractor()

        # Validate input paths early so subsequent operations can rely on them existing.
        if not os.path.exists(arg_input_video_folder):
            raise ValueError("Video folder is not a valid folder: " + arg_input_video_folder)
        if not os.path.exists(arg_user_root_folder):
            raise ValueError("User folder is not a valid folder: " + arg_user_root_folder)

        # Each action is guarded by a membership check so multiple tasks can run in one invocation.
        if ActionOptions.GENERATE_MOTION_FILES.value in arg_selection:
            tmp_motion_start_detector.create_motion_start_files(arg_input_video_folder)

        if ActionOptions.INIT_USERS.value in arg_selection:
            tmp_user_handler.init(arg_user_root_folder)

        if ActionOptions.MATCH_LOCATION_IN_MOTION_FILES.value in arg_selection:
            tmp_motion_time_user_matching.match_motion_start_times_with_users(
                arg_input_video_folder, tmp_user_handler
            )

        if ActionOptions.GENERATE_FULL_CLIP_OUTPUTS.value in arg_selection:
            clip_specification_list = tmp_user_handler.create_clip_specifications(ClipTypes.FULL_CLIP)
            tmp_full_clip_cut_extractor.extract_clips_from_list(clip_specification_list)

        if ActionOptions.GENERATE_TRACKER_CLIP_OUTPUTS.value in arg_selection:
            clip_specification_list = tmp_user_handler.create_clip_specifications(ClipTypes.TRACKER_CLIP)
            tmp_tracker_clip_extractor.extract_clips_from_list(clip_specification_list)
