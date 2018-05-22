import logging, os
import MotionStartDetector, UserHandler, MotionTimeUserMatching

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
        self.motion_time_user_matching = MotionTimeUserMatching.MotionTimeUserMatching()

    def run(self, arg_input_video_folder, arg_user_root_folder):
        logger.info("Running Application with video folder: " + arg_input_video_folder + " user folder: " + arg_user_root_folder + "\n")
        if not os.path.exists(arg_input_video_folder):
            raise ValueError("Video folder is not a valid folder: " + arg_input_video_folder)
        if not os.path.exists(arg_user_root_folder):
            raise ValueError("User folder is not a valid folder: " + arg_user_root_folder)
        self.motion_start_detector.create_motion_start_files(arg_input_video_folder)
        self.user_handler.init(arg_user_root_folder)
        self.motion_time_user_matching.match_motion_start_times_with_users(arg_input_video_folder, self.user_handler)
        clip_specification_list = self.user_handler.create_full_clip_specifications()
        for clip in clip_specification_list:
            print(clip.output_video_path)

