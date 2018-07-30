
import logging, os, cv2, datetime, csv
import numpy as np
import GoproVideo
import ast
from functools import partial
from MotionFilesHandler import MotionFilesHandler
from multiprocessing.pool import Pool
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)

class MotionStartDetector:
    def __init__(self):
        logger.debug("MotionStartDetector created")
        tmp_options = ConfigurationHandler.get_configuration()
        self.number_of_process_workers = int(tmp_options['MOTION_DETECTION']['number_of_process_workers'])

    # Main public function:
    def create_motion_start_files(self, arg_input_video_folder):
        logger.info("Processing input video files which does not have a associated motion file")
        tmp_options = ConfigurationHandler.get_configuration()
        process_video_list = self.__get_list_of_videos_for_processing(arg_input_video_folder)

        process_video_path_list = []
        for video_for_process in process_video_list:
            process_video_path_list.append(video_for_process.path)

        # Single process
        #for video_path in process_video_path_list:
        #    MotionStartDetector.process_video(video_path)

        # multiprocess
        with Pool(processes=self.number_of_process_workers) as pool:
            pool.map(partial(MotionStartDetector.process_video, tmp_options), process_video_path_list)

        return

    def __get_list_of_videos_for_processing(self, arg_input_video_folder):
        tmp_motion_files_handler = MotionFilesHandler()
        existing_motion_files = tmp_motion_files_handler.get_motion_file_list(arg_input_video_folder)
        new_video_files_for_processing = []  # empty list for the video files which has not been processed

        for input_video_file in os.scandir(arg_input_video_folder):
            if input_video_file.name.endswith('.MP4') and input_video_file.is_file():
                if not tmp_motion_files_handler.has_associated_motion_file(existing_motion_files, input_video_file):
                    new_video_files_for_processing.append(input_video_file)
                    logger.debug(input_video_file.name + " is detected as unprocessed file")
        return new_video_files_for_processing #list of files

    @staticmethod
    def find_motion_start_times(arg_option_obj, arg_video_for_process_path):
        logger.info("Finding motion start times for video: " + arg_video_for_process_path)

        tmp_show_video_debug = bool(arg_option_obj['MOTION_DETECTION']['show_video_debug'])
        tmp_morph_open_size = int(arg_option_obj['MOTION_DETECTION']['morph_open_size'])
        tmp_GMG_initializationFrames = int(arg_option_obj['MOTION_DETECTION']['GMG_initializationFrames'])
        tmp_GMG_decisionThreshold = float(arg_option_obj['MOTION_DETECTION']['GMG_decisionThreshold'])
        tmp_allowed_clip_interval = float(arg_option_obj['MOTION_DETECTION']['allowed_clip_interval'])  # [s] Required interval time between valid motion detection
        tmp_motion_frame_counter_threshold = int(arg_option_obj['MOTION_DETECTION']['motion_frame_counter_threshold'])  # required number of frames with movement in mask before making a motion conclusion
        tmp_relative_search_box_dimensions = ast.literal_eval(arg_option_obj['MOTION_DETECTION']['search_box_dimensions'])  # [-] area to look for motion in


        foreground_extractor_GMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=tmp_GMG_initializationFrames, decisionThreshold=tmp_GMG_decisionThreshold)
        morph_open_kernel = np.ones((tmp_morph_open_size, tmp_morph_open_size), np.uint8)

        MyGoproVideo = GoproVideo.GoproVideo(arg_option_obj)
        MyGoproVideo.init(arg_video_for_process_path)
        tmp_absolute_search_box_dimensions = [
            int(tmp_relative_search_box_dimensions[0] * MyGoproVideo.height),
            int(tmp_relative_search_box_dimensions[1] * MyGoproVideo.height),
            int(tmp_relative_search_box_dimensions[2] * MyGoproVideo.width),
            int(tmp_relative_search_box_dimensions[3] * MyGoproVideo.width)
            ]

        next_allowed_motion_frame = 0
        motion_frame_counter = 0
        motion_start_time_list = []

        start_frame = 1
        for iterator in range(start_frame, int(MyGoproVideo.frames)):
            read_return_value, frame, frame_number = MyGoproVideo.read_frame()
            if read_return_value == 0:  # end of file
                logger.debug("End of file: " + arg_video_for_process_path)
                break
            if read_return_value == 20:  # GoPro video error
                continue

            frame_cut = frame[tmp_absolute_search_box_dimensions[0]:tmp_absolute_search_box_dimensions[1], tmp_absolute_search_box_dimensions[2]:tmp_absolute_search_box_dimensions[3]]
            # Resize the frame
            # frame_cut = cv2.resize(frame_cut, None, fx=0.50, fy=0.50, interpolation = cv2.INTER_LINEAR )
            mask = foreground_extractor_GMG.apply(frame_cut)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_open_kernel)

            if mask.any() and frame_number > next_allowed_motion_frame:
                motion_frame_counter += 1
                if motion_frame_counter >= tmp_motion_frame_counter_threshold:
                    motion_frame_counter = 0
                    next_allowed_motion_frame = frame_number + int(MyGoproVideo.fps * tmp_allowed_clip_interval)
                    relative_start_time = datetime.timedelta(seconds=int((frame_number - tmp_motion_frame_counter_threshold) / MyGoproVideo.fps))
                    abs_motion_start_time = MyGoproVideo.creation_time + relative_start_time
                    logger.debug("Motion detected at frame: " + str(frame_number) + ", corrisponding to relative time: " + str(relative_start_time) + ", absolute time: " + abs_motion_start_time.strftime("%Y%m%d_%H_%M_%S"))
                    print("Motion detected at frame: " + str(frame_number) + ", corrisponding to relative time: " + str(relative_start_time) + ", absolute time: " + abs_motion_start_time.strftime("%Y%m%d_%H_%M_%S"))
                    motion_start_time_list.append(abs_motion_start_time)

            if tmp_show_video_debug:
                #cv2.imshow('frame', frame)
                cv2.imshow('frame_cut', frame_cut)
                cv2.imshow('mask', mask)
                cv2.waitKey(1)
            #if len(motion_start_time_list) >= 5:
            #    break

        logger.info("Finished detection motion in file: " + arg_video_for_process_path)
        return motion_start_time_list

    @staticmethod
    def process_video(arg_option_obj, arg_video_for_process_path):
        motion_start_times = MotionStartDetector.find_motion_start_times(arg_option_obj, arg_video_for_process_path)
        MotionFilesHandler.write_motion_file(arg_option_obj, arg_video_for_process_path, motion_start_times)
        return