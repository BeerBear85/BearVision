import fileinput
import re, logging, datetime
import numpy as np
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)

class FullClipSpecification:
    def __init__(self, arg_input_video_file, arg_start_time: datetime, arg_output_video_path: str):
        tmp_options = ConfigurationHandler.get_configuration()
        self.video_file = arg_input_video_file
        self.start_time = arg_start_time + datetime.timedelta(seconds=float(tmp_options['FULL_CLIP_SPECIFICATION']['start_time_offset']))
        self.duration = datetime.timedelta(seconds=float(tmp_options['FULL_CLIP_SPECIFICATION']['clip_duration']))
        self.output_video_relative_speed = float(tmp_options['FULL_CLIP_SPECIFICATION']['output_video_speed'])
        self.output_video_scale = float(tmp_options['FULL_CLIP_SPECIFICATION']['pa_output_scale'])
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
