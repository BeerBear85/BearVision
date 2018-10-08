import logging, os, re, csv, datetime
import pandas as pd
from ConfigurationHandler import ConfigurationHandler


logger = logging.getLogger(__name__)

class MotionFilesHandler:
    def __init__(self):
        return

    @staticmethod
    def has_associated_motion_file(arg_motion_file_list, arg_video_file):
        tmp_options = ConfigurationHandler.get_configuration()
        tmp_motion_file_ending = tmp_options['MOTION_DETECTION']['motion_file_ending']
        logger.debug("Checking if motion file already exists for file: " + arg_video_file.name)
        for motion_file in arg_motion_file_list:  # look through all found motion files for a match
            # logger.debug("Matching: " + motion_file.name + " and " + os.path.splitext(arg_video_file.name)[0] + motion_file_ending + ".csv")
            if motion_file.name == os.path.splitext(arg_video_file.name)[0] + tmp_motion_file_ending + ".csv":
                return True
        return False  #Not match found

    @staticmethod
    def get_associated_video_file(arg_motion_file):
        logger.debug("Checking if associated video file exists for file: " + arg_motion_file.name)
        tmp_options = ConfigurationHandler.get_configuration()
        tmp_motion_file_ending = tmp_options['MOTION_DETECTION']['motion_file_ending']
        motion_file_dir = os.path.dirname(arg_motion_file.path)
        video_file_name = arg_motion_file.name.replace(tmp_motion_file_ending + ".csv", ".MP4")
        for dir_file in os.scandir(motion_file_dir):  # Only look for ass. video file in same dir as the motion file
            #print(dir_file.name + " the same as: " + video_file_name)
            if dir_file.name == video_file_name:
                return dir_file
        return False  # Not match found

    @staticmethod
    def get_motion_file_list(arg_input_video_folder):
        tmp_options = ConfigurationHandler.get_configuration()
        tmp_motion_file_ending = tmp_options['MOTION_DETECTION']['motion_file_ending']
        tmp_motion_file_ending_regex = re.compile(".+" + tmp_motion_file_ending + "\.csv$")
        motion_file_list = []
        for dir_file in os.scandir(arg_input_video_folder):
            # logger.debug("Checking if the following file is a motion start file: " + dir_file.name)
            if tmp_motion_file_ending_regex.match(dir_file.name):
                logger.debug("Motion start file found: " + dir_file.name)
                motion_file_list.append(dir_file)

        return motion_file_list  # list of found motion files

    @staticmethod
    def read_motion_file(arg_motion_file):
        # https://stackoverflow.com/questions/21269399/datetime-dtypes-in-pandas-read-csv
        logger.info("Reading motion file: " + arg_motion_file.path)
        csv_file = open(arg_motion_file, 'r', newline='')
        tmp_motion_start_info = pd.read_csv(csv_file, sep=',', parse_dates=['time'])

        #tmp_motion_start_time = tmp_motion_start_info['time'].tolist()

        return tmp_motion_start_info

    @staticmethod
    def write_motion_file(arg_option_obj, arg_video_file_path, arg_motion_start_info):

        tmp_motion_file_ending = arg_option_obj['MOTION_DETECTION']['motion_file_ending']
        output_filename_short = os.path.splitext(arg_video_file_path)[0] + tmp_motion_file_ending + ".csv"
        output_dir = os.path.dirname(arg_video_file_path)
        output_filename = os.path.join(output_dir, output_filename_short)

        logger.info("Writing motion file: " + output_filename + " with " + str(len(arg_motion_start_info)) + " entries")

        arg_motion_start_info.to_csv(output_filename, index=False)

        logger.debug("Finished writing motion file: " + output_filename)

        return True  # true for success
