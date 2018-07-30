import logging, os
import MotionStartDetector, UserHandler, MotionTimeUserMatching, FullClipExtractor

logger = logging.getLogger(__name__)  #Set logger to reflect the current file

class Application:
    def __init__(self):
        logger.debug("Appliation created")

    def run(self, arg_input_video_folder, arg_user_root_folder, arg_selection):
        logger.info("Running Application with video folder: " + arg_input_video_folder + " user folder: " + arg_user_root_folder + "\n")
        
        #Create objects
        tmp_motion_start_detector = MotionStartDetector.MotionStartDetector()
        tmp_user_handler = UserHandler.UserHandler()
        tmp_motion_time_user_matching = MotionTimeUserMatching.MotionTimeUserMatching()
        tmp_full_clip_cut_extractor = FullClipExtractor.FullClipExtractor()

        if not os.path.exists(arg_input_video_folder):
            raise ValueError("Video folder is not a valid folder: " + arg_input_video_folder)
        if not os.path.exists(arg_user_root_folder):
            raise ValueError("User folder is not a valid folder: " + arg_user_root_folder)

        if 0 in arg_selection:
            tmp_motion_start_detector.create_motion_start_files(arg_input_video_folder)

        if 1 in arg_selection:
            tmp_user_handler.init(arg_user_root_folder)

        if 2 in arg_selection:
            tmp_motion_time_user_matching.match_motion_start_times_with_users(arg_input_video_folder, tmp_user_handler)

        if 3 in arg_selection:
            clip_specification_list = tmp_user_handler.create_full_clip_specifications()
            tmp_full_clip_cut_extractor.extract_full_clip_specifications(clip_specification_list)
