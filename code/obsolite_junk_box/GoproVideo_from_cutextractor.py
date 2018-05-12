# File: GoproVideo.py
# A class for getting the info of a GoPro videofile

import re, cv2, subprocess
import datetime as dt
#from subprocess import call

class GoproVideo:
	def __init__(self):
		self.videoreader_obj = 0
		self.old_filename = ""
		self.creation_time   = 0 #datetime
		self.width  = 0
		self.height = 0
		self.fps    = 0
		self.frames = 0 #total number of frames
		
	def init(self, arg_specfile_obj):
		if (arg_specfile_obj.video_filename != self.old_filename):
			if self.videoreader_obj != 0:
				self.videoreader_obj.release()
			self.videoreader_obj = cv2.VideoCapture(arg_specfile_obj.video_filename)
			self.extract_video_spec()
			self.extract_creation_date(arg_specfile_obj.video_filename)
			self.old_filename = arg_specfile_obj.video_filename
		
		
	def extract_creation_date(self, arg_filename):
		cmd_line = ['ffprobe', '-show_format', '-pretty', '-loglevel', 'quiet', arg_filename]
		ffprobe_process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err =  ffprobe_process.communicate()
		if err:
			print("========= error ========")
			print(err)
		out_string = out.decode('UTF-8');
		create_time_str = re.findall("(?<=creation_time=)[\S\ \-\:]+", out_string)[0]
		#print(create_time_str)
		self.creation_time = dt.datetime.strptime(create_time_str, "%Y-%m-%d %H:%M:%S")
		
	def extract_video_spec(self):
		self.width  = self.videoreader_obj.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.videoreader_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.fps    = self.videoreader_obj.get(cv2.CAP_PROP_FPS)
		self.frames = self.videoreader_obj.get(cv2.CAP_PROP_FRAME_COUNT)
		return
		
	def set_start_point(self, arg_start_frame):
		self.videoreader_obj.set(cv2.CAP_PROP_POS_FRAMES, arg_start_frame)
		return
		
	def read_frame(self):
		read_return_value, frame = self.videoreader_obj.read()
		return read_return_value, frame
	