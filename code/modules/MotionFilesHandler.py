import logging, os


logger = logging.getLogger(__name__)


class MotionFilesHandler:
    def __init__(self):
        return

    def check_for_unprocessed_files(self, arg_input_video_folder):
        return  #list of files

    def get_motion_file_list(self, arg_input_video_folder):
        return  #list of found motion files

    def read_motion_file(self, arg_motion_file_filename):
        return  #list of datetimes

    def write_motion_file(self, arg_motion_times_list):
        return  #true for succes
