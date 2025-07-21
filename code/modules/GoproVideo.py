# File: GoproVideo.py
# A class for getting the info of a GoPro videofile
# It can do things as extract the start time from the GPS meta data, set the starting point of the video at a specific frame and read a frame of the video

import re, cv2, subprocess, os, logging, warnings, tempfile
import datetime as dt
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)

class GoproVideo:
    def __init__(self, arg_options_obj = None):
        self.videoreader_obj = 0
        self.current_filename = ""
        self.creation_time = 0  # datetime
        self.width = 0
        self.height = 0
        self.fps = 0
        self.frames = 0  # total number of frames
        self.current_frame = 0
        #if arg_options_obj is None:
        #   arg_options_obj = ConfigurationHandler.get_configuration()
        #self.tool_folder = arg_options_obj['GOPRO_VIDEO']['tool_folder']
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.tool_folder = os.path.abspath(os.path.join(current_dir, '..', '..', 'tools'))

    def init(self, arg_video_filename):
        if (arg_video_filename != self.current_filename):  # Only initilise if it is a new file
            self.current_filename = arg_video_filename
            if self.videoreader_obj != 0:
                self.videoreader_obj.release()
            self.videoreader_obj = cv2.VideoCapture(arg_video_filename)
            self.extract_video_spec()
            self.extract_creation_date()
            self.current_frame = 1

    def extract_creation_date(self):
        return_code = self.extract_creation_date_from_file_gps()
        if return_code != 0:
            warnings.warn("No GPS info was found. Extracting the recoding time using non-GPS info instead")
            self.extract_creation_date_from_file_info()

    def extract_creation_date_from_file_gps(self):
        # Use ffmpeg to extract metadata stream (GoPro MET) - stream number 3
        ffmpeg_path = os.path.join(self.tool_folder, 'ffmpeg')

        (temp_bin_file_handle, temp_bin_file_name) = tempfile.mkstemp(prefix="gopro_temp_", suffix=".bin")
        (temp_json_file_handle, temp_json_file_name) = tempfile.mkstemp(prefix="gopro_temp_", suffix=".json")

        cmd_line = [ffmpeg_path, '-y', '-i', self.current_filename, '-loglevel', 'error', '-codec', 'copy', '-map',
                    '0:3', '-f', 'rawvideo', temp_bin_file_name]
        logger.debug("Calling command for .bin gen: " + ' '.join(cmd_line))
        process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            print("========= error ========")
            print(err.decode('UTF-8'))
        # return -1;
        out_string = out.decode('UTF-8')
        logger.debug('Output: %s', out_string)

        # Use tool to convert metadata to json format
        gopro2json_path = os.path.join(self.tool_folder, 'gopro2json')
        cmd_line = [gopro2json_path, '-i', temp_bin_file_name, '-o', temp_json_file_name]
        logger.debug("Calling command for json gen: %s", ' '.join(cmd_line))
        process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            print("========= error ========")
            print(err.decode('UTF-8'))
            raise ValueError('Something went wrong when extracting meta data from GoPro video!')
        # return -1;
        out_string = out.decode('UTF-8')
        logger.debug('Output: ' + out_string)

        # Clean up
        if os.path.isfile(temp_bin_file_name):
            os.close(temp_bin_file_handle)
            os.unlink(temp_bin_file_name)

        # Read .json file and extract first GPS timestamp
        if os.access(temp_json_file_name, os.R_OK):
            textfile = open(temp_json_file_name, 'r')
            filetext = textfile.read()
            textfile.close()
            utc_time_match_list = re.findall("(?<=\"utc\":)\d+", filetext)
            if utc_time_match_list:
                utc_time = int(utc_time_match_list[0])
            else:
                logger.debug("Did not find any UTC time stamp")
                return -1

            logger.debug("Found UTC time: %s", utc_time)
            self.creation_time = dt.datetime.utcfromtimestamp(utc_time / 1000000)  # Devide by 1000000 to match the format of datetime
            logger.debug("Converted creation time: %s", self.creation_time.strftime("%Y%m%d_%H_%M_%S_%f"))

            # Clean up
            if os.path.isfile(temp_json_file_name):
                os.close(temp_json_file_handle)
                os.unlink(temp_json_file_name)
        else:
            logger.debug("Could not open the json file")
            return -1

        return 0

    # Get creation time from file info
    def extract_creation_date_from_file_info(self):
        ffprobe_path = os.path.join(self.tool_folder, 'ffprobe')
        cmd_line = [ffprobe_path, '-show_format', '-pretty', '-loglevel', 'quiet', self.current_filename]
        logger.debug("Calling command for ffprobe file info: " + ' '.join(cmd_line))
        ffprobe_process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffprobe_process.communicate()
        if err:
            print("========= error ========")
            print(err)
        # return -1
        out_string = out.decode('UTF-8')
        create_time_str = re.findall("(?<=creation_time=)[\S\ \-\:]+", out_string)[0]
        # print(create_time_str)
        self.creation_time = dt.datetime.strptime(create_time_str, "%Y-%m-%d %H:%M:%S")

    def extract_video_spec(self):
        self.width = int(self.videoreader_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.videoreader_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.videoreader_obj.get(cv2.CAP_PROP_FPS)
        self.frames = int(self.videoreader_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        return

    def set_start_point(self, arg_start_frame):
        self.videoreader_obj.set(cv2.CAP_PROP_POS_FRAMES, arg_start_frame)
        self.current_frame = arg_start_frame
        return

    def get_relative_time(self):
        """Get the relative time in milliseconds from the start of the video"""
        rel_time_ms = self.videoreader_obj.get(cv2.CAP_PROP_POS_MSEC)
        return rel_time_ms

    # Consider using VideoStream for increased speed
    def read_frame(self):
        """ Read a frame from the video and return it along with the frame number"""
        read_return_value, frame = self.videoreader_obj.read()
        self.current_frame += 1

        if (read_return_value == 0) & (self.current_frame < self.frames):
            logger.debug("Missed frame: " + str(self.current_frame) + " now skipping a few frames ahead")
            self.current_frame += 5
            self.set_start_point(self.current_frame)
            read_return_value = 20

        return read_return_value, frame, self.current_frame

    def export_video_part(self, arg_output_filename, arg_rel_start_time, arg_duration):
        # ffmpeg -i [input_file] -ss [start_seconds] -t [duration_seconds] [output_file]
        start_str = str(arg_rel_start_time)  # start_time.strftime("%H:%M:%S")
        duration_str = str(arg_duration)  # duration.strftime("%H:%M:%S")
        ffmpeg_path = os.path.join(self.tool_folder, 'ffmpeg')
        cmd_line = [ffmpeg_path, '-i', self.current_filename, '-ss', start_str, '-t', duration_str, '-vcodec', 'copy',
                    '-acodec', 'copy', arg_output_filename]
        print(cmd_line)
        ffmpeg_process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg_process.communicate()
        if err:
            # print("========= error ========")
            aaa = err
        #	print(err)
        return
