import logging, os, re


logger = logging.getLogger(__name__)

motion_file_ending = "_motion_start_times"
motion_file_ending_regex = re.compile(".+" + motion_file_ending + "\.csv$")


class MotionFilesHandler:
    def __init__(self):
        return

    @staticmethod
    def has_associated_motion_file(arg_motion_file_list, arg_video_file):
        logger.debug("Checking if motion file already exists for file: " + arg_video_file.name)
        for motion_file in arg_motion_file_list:  # look through all found motion files for a match
            #logger.debug("Matching: " + motion_file.name + " and " + os.path.splitext(arg_video_file.name)[0] + motion_file_ending + ".csv")
            if motion_file.name == os.path.splitext(arg_video_file.name)[0] + motion_file_ending + ".csv":
                return True

        return False  #Not match found

    @staticmethod
    def get_motion_file_list(arg_input_video_folder):
        motion_file_list = []
        for dir_file in os.scandir(arg_input_video_folder):
            #logger.debug("Checking if the following file is a motion start file: " + dir_file.name)
            if motion_file_ending_regex.match(dir_file.name):
                logger.debug("Motion start file found: " + dir_file.name)
                motion_file_list.append(dir_file)

        return motion_file_list  #list of found motion files

    def read_motion_file(self, arg_motion_file_filename):
        return  #list of datetimes

    def write_motion_file(self, arg_motion_times_list):
        return  #true for succes
