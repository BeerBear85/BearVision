import fileinput
import re, logging, datetime
import numpy as np
logger = logging.getLogger(__name__)


pa_clip_duration = 6  # [s]
pa_start_time_offset = 0.5  # [s]
pa_output_video_speed = 0.5  # 0.5 means that a video clip output should be half speed of input file
pa_output_scale = 0.5  # 0.5 means half the height/width of the original video

class FullClipSpecification:
    def __init__(self, arg_input_video_file, arg_start_time: datetime, arg_output_video_path: str):
      
        self.video_file = arg_input_video_file
        self.start_time = arg_start_time + datetime.timedelta(seconds=pa_start_time_offset)
        self.duration = datetime.timedelta(seconds=pa_clip_duration)
        self.output_video_relative_speed = pa_output_video_speed
        self.output_video_scale = pa_output_scale
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
