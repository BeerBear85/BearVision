import logging, os
from UserHandler import UserHandler
from MotionFilesHandler import MotionFilesHandler
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)

class MotionTimeUserMatching:
    def __init__(self):
        tmp_options = ConfigurationHandler.get_configuration()
        self.obstacle_approach_location = [float(tmp_options['OBSTACLE']['approach_location_lat']), float(tmp_options['OBSTACLE']['approach_location_long'])]
        self.user_match_minimum_interval = float(tmp_options['OBSTACLE']['user_match_minimum_interval'])
        return

    # Updates all the users database of known matches
    def match_motion_start_times_with_users(self, arg_input_video_folder, arg_user_handler: UserHandler):
        motion_times_files = MotionFilesHandler.get_motion_file_list(arg_input_video_folder)

        for motion_file in motion_times_files:
            motion_start_times_list = MotionFilesHandler.read_motion_file(motion_file)
            for start_time_entry in motion_start_times_list:
                user = arg_user_handler.find_valid_user_match(start_time_entry, self.obstacle_approach_location)
                if user != 0:
                    #print("User match found: " + user.name)
                    video_file = MotionFilesHandler.get_associated_video_file(motion_file)
                    if video_file:  # The video file exists
                        user.add_obstacle_match(start_time_entry, video_file) # Add info to user data
                        logger.debug("Adding match entry to user: " + user.name)
                    else:
                        logger.warning("No associated video file found for motion file: " + motion_file.path)

        arg_user_handler.filter_obstacle_matches(self.user_match_minimum_interval)
        return
