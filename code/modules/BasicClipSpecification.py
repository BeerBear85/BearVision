import fileinput
import re, logging, datetime
import numpy as np
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)

class BasicClipSpecification:
    def __init__(self, arg_input_video_file_path: str, arg_start_time: datetime, arg_output_video_path: str):
        logger.debug("Created full clip spec for " + arg_output_video_path + " from video file: " + arg_input_video_file_path)
        tmp_options = ConfigurationHandler.get_configuration()
        self.video_file_path = arg_input_video_file_path
        self.start_time = arg_start_time + datetime.timedelta(seconds=float(tmp_options['FULL_CLIP_SPECIFICATION']['start_time_offset']))
        self.output_video_path = arg_output_video_path

    def read_file(self, arg_full_clip_spec_filename):
        logger.error("read_file - Not implemented yet!")

    def write_file(self, arg_full_clip_spec_filename):
        logger.error("write_file - Not implemented yet!")


# Notes for CutSpecificationFile:
        # # Consider using module "configparser"
        # textfile = open(self.spec_filename, 'r') #maybe require "rb"
        # filetext = textfile.read()
        # textfile.close()
        #
        # self.video_filename = re.findall("(?<=Video filename: )[\S\ ]+", filetext)[0]
        # self.track_id       = int(re.findall("(?<=Track id: )\S+", filetext)[0])
        # self.start_frame    = int(re.findall("(?<=Start frame: )\S+", filetext)[0])
        # self.track_age      = int(re.findall("(?<=Track age: )\S+", filetext)[0])
        # self.box_dimensions = np.fromstring(re.findall("(?<=Bounding box dimension: )\S+", filetext)[0], dtype=int, sep=',')
        #
        # #Read coordinate vector in the bottom of the file
        # textfile = open(self.spec_filename, 'rb')
        # self.box_coordinates = np.loadtxt(textfile, dtype=int, delimiter=",", skiprows=6)
