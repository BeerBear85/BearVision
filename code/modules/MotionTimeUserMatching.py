import logging, os
import UserHandler
from MotionFilesHandler import MotionFilesHandler

logger = logging.getLogger(__name__)


class MotionTimeUserMatching:
    def __init__(self):
        return

    def match_motion_start_times_with_users(self, arg_input_video_folder, arg_user_handler):
        motion_times_files = MotionFilesHandler.get_motion_file_list(arg_input_video_folder)

        for motion_file in motion_times_files:
            motion_start_times_list = MotionFilesHandler.read_motion_file(motion_file)

        return
