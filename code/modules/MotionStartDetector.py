
import logging, glob, re, os, cv2
import GoproVideo

tmp_show_video = True

motion_file_ending = "_motion_start_times"

class MotionStartDetector:
    def __init__(self):
        logging.debug("MotionStartDetector created\n")
        self.MyGoproVideo = GoproVideo.GoproVideo()

    def Init(self):
        logging.debug("Init called\n")

    def CreateMotionStartFiles(self, arg_input_video_folder):
        process_video_list = self.GetListOfVideosForProcessing(arg_input_video_folder)

        for video_for_process in process_video_list:
            motion_start_times = self.FindMotionStartTimes(video_for_process)

    def GetListOfVideosForProcessing(self, arg_input_video_folder):
        logging.debug("CreateMotionStartFile start\n")
        
        input_video_dir_files = os.scandir(arg_input_video_folder) # - other way of lokking in dir
        input_video_dir_files_copy = os.scandir(arg_input_video_folder)
        
        input_video_files = [item for item in input_video_dir_files if os.path.splitext(item)[1] == ".MP4"]
        existing_motion_files = [item for item in input_video_dir_files_copy if os.path.splitext(item)[1] == ".csv"]
        
        #input_video_list          = glob.glob(arg_input_video_folder + "/*.mp4", recursive=True)
        #existing_motion_file_list = glob.glob(arg_input_video_folder + "/*motion_start_times.csv", recursive=True)
        
        new_video_files_for_processing = [] #empty list for the video files which has not been processed
            
        for input_video_file in input_video_files:
            if input_video_file.name.endswith('.MP4') and input_video_file.is_file():
                logging.debug("Checking file: " + input_video_file.name)
                motion_file_exists = False
                for motion_file in existing_motion_files: #look through all found motion files for a match
                    if (motion_file.name == os.path.splitext(input_video_file.name)[0] + motion_file_ending + ".csv"):
                        logging.debug(input_video_file.name + " motion file already exists!")
                        motion_file_exists = True
                if motion_file_exists == False:
                    new_video_files_for_processing.append(input_video_file)

        logging.debug("New video files for motion detection processing:")
        for input_video_file in new_video_files_for_processing:
            logging.debug(input_video_file.name)
                        
        return new_video_files_for_processing

    def FindMotionStartTimes(self, arg_video_for_process):
        logging.debug("Finding motion start times for video: " + arg_video_for_process.path)
        self.MyGoproVideo.init(arg_video_for_process.path)

        start_frame = 1
        for frame_number in range(start_frame, int(self.MyGoproVideo.frames)):
            read_return_value, frame = self.MyGoproVideo.read_frame()
            if (read_return_value == 0):  # end of file
                print('End of file!!!')
                break
            cv2.imshow('frame', frame)
            cv2.waitKey(50)

