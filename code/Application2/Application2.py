import logging
import MotionStartDetector
import glob


logging.basicConfig(filename='debug.log',level=logging.DEBUG)

class Application:
    def __init__(self):
        logging.debug("Appliation created\n")
        self.motion_start_detector = MotionStartDetector.MotionStartDetector()

    def Init(self):
        logging.debug("Init called\n")

    def run(self, arg_input_video_folder, arg_user_folder):
        logging.debug("Running Application with video folder: " + arg_input_video_folder + " user folder: " + arg_user_folder + "\n")
        self.motion_start_detector.CreateMotionStartFile(arg_input_video_folder)