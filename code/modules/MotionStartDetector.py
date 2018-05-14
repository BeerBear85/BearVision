
import logging, glob
#logging.basicConfig(filename='debug2.log', level=logging.DEBUG)

tmp_show_cideo = True

class MotionStartDetector:
    def __init__(self):
        logging.debug("MotionStartDetector created\n")

    def Init(self):
        logging.debug("Init called\n")

    def CreateMotionStartFiles(self, arg_input_video_folder):
        logging.debug("CreateMotionStartFile start\n")

        input_video_list          = glob.glob(arg_input_video_folder + "/*.mp4", recursive=True)
        existing_motion_file_list = glob.glob(arg_input_video_folder + "/*motion_start_times.csv", recursive=True)

        for input_video_file in input_video_list
            # TODO check if input_video file already have a matching .csv file
            logging.info(existing_motion_file_list)
