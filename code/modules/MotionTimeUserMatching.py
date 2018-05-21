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
                    video_file = MotionFilesHandler.get_associated_video_file(motion_file)
                    if video_file:  # The video file exists
                        user.add_obstacle_match(start_time_entry, video_file) # Add info to user data
                        logger.debug("Adding match entry to user: " + user.name)
                    else:
                        logger.warning("No associated video file found for motion file: " + motion_file.path)

        #TODO Only for initial test!
        arg_user_handler.user_list[1].obstacle_match_data.to_csv('db_output.csv', index=False, header=False)
        return
