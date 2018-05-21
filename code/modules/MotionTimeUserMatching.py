import logging, os
import UserHandler
from MotionFilesHandler import MotionFilesHandler

logger = logging.getLogger(__name__)

obstacle_approach_location = [55.682366, 12.623255]  # lat/lon #This info should come from an obstacle_info file


class MotionTimeUserMatching:
    def __init__(self):
        return

    def match_motion_start_times_with_users(self, arg_input_video_folder, arg_user_handler):
        motion_times_files = MotionFilesHandler.get_motion_file_list(arg_input_video_folder)

        for motion_file in motion_times_files:
            motion_start_times_list = MotionFilesHandler.read_motion_file(motion_file)
            for start_time_entry in motion_start_times_list:
                user = arg_user_handler.find_valid_user_match(start_time_entry, obstacle_approach_location)
                if user != 0:
                    print("User match found: " + user.name)

        return
