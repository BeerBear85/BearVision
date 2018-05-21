
import logging, os, cv2, datetime, csv
import numpy as np
import GoproVideo
from MotionFilesHandler import MotionFilesHandler

logger = logging.getLogger(__name__)
tmp_show_video_debug = False

morph_open_size = 20
GMG_initializationFrames = 60
GMG_decisionThreshold = 0.8
#frame_cut_dimensions = [650, 950, 1, 300]  #[Pix] area to look for motion in
frame_cut_dimensions = [600, 1000, 1, 300]  #[Pix] area to look for motion in
allowed_clip_interval = 5            #[s] Required interval time between valid motion detection
start_caption_offset = 0.5           #[s] rewind offset for when capturing clip
motion_frame_counter_threshold = 3   #required number of frames with movement in mask before making a motion conclusion

class MotionStartDetector:
    def __init__(self):
        logger.debug("MotionStartDetector created")
        self.MyGoproVideo = GoproVideo.GoproVideo()
        self.foreground_extractor_GMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=GMG_initializationFrames, decisionThreshold=GMG_decisionThreshold)
        self.morph_open_kernel = np.ones((morph_open_size,morph_open_size),np.uint8)

    #Main public function:
    def create_motion_start_files(self, arg_input_video_folder):
        logger.debug("create_motion_start_files() start")
        process_video_list = self.__get_list_of_videos_for_processing(arg_input_video_folder)

        for video_for_process in process_video_list:
            motion_start_times = self.__find_motion_start_times(video_for_process)
            MotionFilesHandler.write_motion_file(video_for_process, motion_start_times)

        return

    def __get_list_of_videos_for_processing(self, arg_input_video_folder):
        existing_motion_files = MotionFilesHandler.get_motion_file_list(arg_input_video_folder)
        new_video_files_for_processing = []  # empty list for the video files which has not been processed

        for input_video_file in os.scandir(arg_input_video_folder):
            if input_video_file.name.endswith('.MP4') and input_video_file.is_file():
                if not MotionFilesHandler.has_associated_motion_file(existing_motion_files, input_video_file):
                    new_video_files_for_processing.append(input_video_file)
                    logger.debug(input_video_file.name + " is detected as unprocessed file")
        return new_video_files_for_processing #list of files
                        


    def __find_motion_start_times(self, arg_video_for_process):
        logger.info("Finding motion start times for video: " + arg_video_for_process.path)
        self.MyGoproVideo.init(arg_video_for_process.path)
        next_allowed_motion_frame = 0
        motion_frame_counter = 0
        motion_start_time_list = []

        start_frame = 1
        for iterator in range(start_frame, int(self.MyGoproVideo.frames)):
            read_return_value, frame, frame_number = self.MyGoproVideo.read_frame()
            if read_return_value == 0:  # end of file
                logger.debug("End of file: " + arg_video_for_process.path)
                break
            if read_return_value == 20:  # GoPro video error
                continue

            frame_cut = frame[frame_cut_dimensions[0]:frame_cut_dimensions[1], frame_cut_dimensions[2]:frame_cut_dimensions[3]]
            # Resize the frame
            # frame_cut = cv2.resize(frame_cut, None, fx=0.50, fy=0.50, interpolation = cv2.INTER_LINEAR )
            mask = self.foreground_extractor_GMG.apply(frame_cut)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_open_kernel)

            if mask.any() and frame_number > next_allowed_motion_frame:
                motion_frame_counter += 1
                if motion_frame_counter >= motion_frame_counter_threshold:
                    motion_frame_counter = 0
                    next_allowed_motion_frame = frame_number + int(self.MyGoproVideo.fps * allowed_clip_interval)
                    relative_start_time = datetime.timedelta(seconds=int((frame_number - motion_frame_counter_threshold) / self.MyGoproVideo.fps))
                    abs_motion_start_time = self.MyGoproVideo.creation_time + relative_start_time
                    logger.debug("Motion detected at frame: " + str(frame_number) + ", corrisponding to relative time: " + str(relative_start_time) + ", absolute time: " + abs_motion_start_time.strftime("%Y%m%d_%H_%M_%S"))
                    motion_start_time_list.append(abs_motion_start_time)

            if tmp_show_video_debug:
                #cv2.imshow('frame', frame)
                cv2.imshow('frame_cut', frame_cut)
                cv2.imshow('mask', mask)
                cv2.waitKey(1)
            #if len(motion_start_time_list) >= 5:
            #    break

        logger.info("Finished detection motion in file: " + arg_video_for_process.path)
        return motion_start_time_list
