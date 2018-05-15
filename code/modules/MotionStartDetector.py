
import logging, glob, re, os
#logging.basicConfig(filename='debug2.log', level=logging.DEBUG)

tmp_show_cideo = True

class MotionStartDetector:
    def __init__(self):
        logging.debug("MotionStartDetector created\n")

    def Init(self):
        logging.debug("Init called\n")

    def CreateMotionStartFiles(self, arg_input_video_folder):
        logging.debug("CreateMotionStartFile start\n")
        
        input_video_dir_files = os.scandir(arg_input_video_folder) # - other way of lokking in dir
        input_video_dir_files_copy = os.scandir(arg_input_video_folder)
        
        input_video_files = [f for f in input_video_dir_files if os.path.splitext(f)[1] == ".mp4"]
        existing_motion_files = [f for f in input_video_dir_files_copy if os.path.splitext(f)[1] == ".csv"]
        
        #input_video_list          = glob.glob(arg_input_video_folder + "/*.mp4", recursive=True)
        #existing_motion_file_list = glob.glob(arg_input_video_folder + "/*motion_start_times.csv", recursive=True)
        
        #logging.debug(input_video_files.name)
        #logging.debug(existing_motion_files.name)
        
        new_video_files_for_processing = [] #empty list for the video files which has not been processed
            
        for input_video_file in input_video_files:
            if input_video_file.name.endswith('.mp4') and input_video_file.is_file():
                logging.debug("Checking file: " + input_video_file.name)
                for motion_file in existing_motion_files:
                    if (motion_file.name != os.path.splitext(input_video_file.name)[0] + "_motion_start_times.csv"):
                        print(input_video_file.name + " motion file does not exists!")
                        new_video_files_for_processing.append(input_video_file)
                        #test = re.findall("\S*_motion_start_times", motion_file.name) #returns list of matches
                        
                        
                        
        for input_video_file in new_video_files_for_processing:
            print(input_video_file.name)
                        
