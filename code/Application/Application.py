import logging
import MotionStartDetector
import UserHandler


logging.basicConfig(filename='debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)-8s:%(name)s:%(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)  #Set logger to reflect the current file

class Application:
    def __init__(self):
        logger.debug("Appliation created")
        self.motion_start_detector = MotionStartDetector.MotionStartDetector()
        self.user_handler = UserHandler.UserHandler()

    def run(self, arg_input_video_folder, arg_user_root_folder):
        logger.debug("Running Application with video folder: " + arg_input_video_folder + " user folder: " + arg_user_root_folder + "\n")
        self.motion_start_detector.create_motion_start_files(arg_input_video_folder)
        self.user_handler.init(arg_user_root_folder)

