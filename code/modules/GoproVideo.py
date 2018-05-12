# File: GoproVideo.py
# A class for getting the info of a GoPro videofile

import re, cv2, subprocess, os, logging
import datetime as dt
#from subprocess import call
tool_folder = os.path.join('..','tools')

class GoproVideo:
	def __init__(self):
		self.videoreader_obj = 0
		self.current_filename = ""
		self.creation_time   = 0 #datetime
		self.width  = 0
		self.height = 0
		self.fps    = 0
		self.frames = 0 #total number of frames
		
	def init(self, arg_video_filename):
		if (arg_video_filename != self.current_filename): #Only initilise if it is a new file
			self.current_filename = arg_video_filename
			if self.videoreader_obj != 0:
				self.videoreader_obj.release()
			self.videoreader_obj = cv2.VideoCapture(arg_video_filename)
			self.extract_video_spec()
			self.extract_creation_date()
		
	def extract_creation_date(self):
		return_code = self.extract_creation_date_from_file_gps()
		if (return_code is not 0):
			warnings.warn("No GPS info was found. Extracting the recoding time using non-GPS info instead")
			extract_creation_date_from_file_info()
		
	def extract_creation_date_from_file_gps(self):
		# Use ffmpeg to extract metadata stream (GoPro MET) - stream number 3
		ffmpeg_path = os.path.join(tool_folder, 'ffmpeg')
		#cmd_line = [ffmpeg_path, '-y', '-i', self.current_filename, '-codec', 'copy', '-map', '0:m:handler_name:\"\tGoPro MET\"', '-f', 'rawvideo', 'temp.bin']
		cmd_line = [ffmpeg_path, '-y', '-i', self.current_filename, '-loglevel', 'error', '-codec', 'copy', '-map', '0:3', '-f', 'rawvideo', 'temp.bin']
		logging.info("Calling command: " + str(cmd_line))
		process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err =  process.communicate()
		if err:
			print("========= error ========")
			print(err.decode('UTF-8'))
			#return -1;
		out_string = out.decode('UTF-8');
		logging.info('Output: ' + out_string)
		
		# Use tool to convert metadata to json format
		gopro2json_path = os.path.join(tool_folder, 'gopro2json')
		cmd_line = [gopro2json_path, '-i', 'temp.bin', '-o', 'temp.json']
		logging.info("Calling command: " + str(cmd_line))
		process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err =  process.communicate()
		if err:
			print("========= error ========")
			print(err.decode('UTF-8'))
			#return -1;
		out_string = out.decode('UTF-8');
		logging.info('Output: ' + out_string)
		os.remove("temp.bin")
		
		#Read .json file and extract first GPS timestamp
		if os.access('temp.json', os.R_OK):
			textfile = open('temp.json', 'r')
			filetext = textfile.read()
			textfile.close()
			utc_time = int(re.findall("(?<=\"utc\":)\d+", filetext)[0])
			
			print("Found UTC time: " + str(utc_time))
			self.creation_time = dt.datetime.utcfromtimestamp(utc_time/1000000) #Devide by 1000000 to match the format of datetime
			os.remove("temp.json")
		else:
			return -1
		
		return 0
		
	#Get creation time from file info
	def extract_creation_date_from_file_info(self):
		ffprobe_path = os.path.join(tool_folder, 'ffprobe')
		cmd_line = [ffprobe_path, '-show_format', '-pretty', '-loglevel', 'quiet', self.current_filename]
		ffprobe_process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = ffprobe_process.communicate()
		if err:
			print("========= error ========")
			print(err)
			#return -1;
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
	
	def export_video_part(self, arg_output_filename, arg_rel_start_time, arg_duration):
		#ffmpeg -i [input_file] -ss [start_seconds] -t [duration_seconds] [output_file]
		start_str = str(arg_rel_start_time) #start_time.strftime("%H:%M:%S")
		duration_str = str(arg_duration) #duration.strftime("%H:%M:%S")
		ffmpeg_path = os.path.join(tool_folder, 'ffmpeg')
		cmd_line = [ffmpeg_path, '-i', self.current_filename, '-ss', start_str, '-t', duration_str, '-vcodec', 'copy', '-acodec', 'copy', arg_output_filename]
		print(cmd_line)
		ffmpeg_process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err =  ffmpeg_process.communicate()
		if err:
			#print("========= error ========")
			aaa = err
		#	print(err)
		return