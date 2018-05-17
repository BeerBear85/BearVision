
import logging, glob, re, os, cv2, datetime, csv
import numpy as np
import GoproVideo

logger = logging.getLogger(__name__)
tmp_show_video = True

motion_file_ending = "_motion_start_times"
morph_open_size = 20
GMG_initializationFrames = 60
GMG_decisionThreshold = 0.8
allowed_clip_interval = 5            #[s] Required interval time between valid motion detection
start_caption_offset = 0.5           #[s] rewind offset for when capturing clip
motion_frame_counter_threshold = 3   #required number of frames with movement in mask before making a motion conclusion

class MotionStartDetector:
    def __init__(self):
        logger.debug("MotionStartDetector created\n")
        self.MyGoproVideo = GoproVideo.GoproVideo()
        self.foreground_extractor_GMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=GMG_initializationFrames, decisionThreshold=GMG_decisionThreshold)
        self.morph_open_kernel = np.ones((morph_open_size,morph_open_size),np.uint8)

    #Main public function:
    def CreateMotionStartFiles(self, arg_input_video_folder):
        process_video_list = self.__get_list_of_videos_for_processing(arg_input_video_folder)

        for video_for_process in process_video_list:
            motion_start_times = self.__find_motion_start_times(video_for_process)
            self.__write_motion_file(video_for_process, motion_start_times)

        return

    def __get_list_of_videos_for_processing(self, arg_input_video_folder):
        logger.debug("CreateMotionStartFile start\n")
        
        input_video_dir_files = os.scandir(arg_input_video_folder) # - other way of lokking in dir
        input_video_dir_files_copy = os.scandir(arg_input_video_folder)
        
        input_video_files = [item for item in input_video_dir_files if os.path.splitext(item)[1] == ".MP4"]
        existing_motion_files = [item for item in input_video_dir_files_copy if os.path.splitext(item)[1] == ".csv"]
        
        #input_video_list          = glob.glob(arg_input_video_folder + "/*.mp4", recursive=True)
        #existing_motion_file_list = glob.glob(arg_input_video_folder + "/*motion_start_times.csv", recursive=True)
        
        new_video_files_for_processing = [] #empty list for the video files which has not been processed
            
        for input_video_file in input_video_files:
            if input_video_file.name.endswith('.MP4') and input_video_file.is_file():
                logger.debug("Checking file: " + input_video_file.name)
                motion_file_exists = False
                for motion_file in existing_motion_files: #look through all found motion files for a match
                    if (motion_file.name == os.path.splitext(input_video_file.name)[0] + motion_file_ending + ".csv"):
                        logger.debug(input_video_file.name + " motion file already exists!")
                        motion_file_exists = True
                if motion_file_exists == False:
                    new_video_files_for_processing.append(input_video_file)

        logger.debug("New video files for motion detection processing:")
        for input_video_file in new_video_files_for_processing:
            logger.debug(input_video_file.name)
                        
        return new_video_files_for_processing

    def __find_motion_start_times(self, arg_video_for_process):
        logger.debug("Finding motion start times for video: " + arg_video_for_process.path)
        self.MyGoproVideo.init(arg_video_for_process.path)
        next_allowed_motion_frame = 0
        motion_frame_counter = 0
        motion_start_time_list = []

        start_frame = 1
        for frame_number in range(start_frame, int(self.MyGoproVideo.frames)):
            read_return_value, frame = self.MyGoproVideo.read_frame()
            if read_return_value == 0:  # end of file
                print('End of file!!!')
                break
            if read_return_value == 20:  # GoPro vido error
                print('Skipping frame: ' + str(frame_number))
                continue

            frame_cut = frame[650:950, 1:300] #TODO should be parameters
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
                    logger.debug("Motion detected at frame: " + str(frame_number))
                    logger.debug("Corrisponding to relative time: " + str(relative_start_time))
                    #abs_motion_start_time = self.MyGoproVideo.creation_time + relative_start_time
                    motion_start_time_list.append(relative_start_time)

            #cv2.imshow('frame', frame)
            #cv2.imshow('frame_cut', frame_cut)
            #cv2.imshow('mask', mask)
            #cv2.waitKey(1)
            #if len(motion_start_time_list) >= 5:
            #    break

        logger.debug("Finished detection motion in file: " + arg_video_for_process.path)
        return motion_start_time_list

    def __write_motion_file(self, arg_video_file, arg_motion_start_times_list):
        output_filename_short = os.path.splitext(arg_video_file.name)[0] + motion_file_ending + ".csv"
        output_filename = os.path.join(os.path.dirname(arg_video_file.path), output_filename_short)

        logger.debug("Writing motion file: " + output_filename + "with " + str(len(arg_motion_start_times_list)) + " entries")
        csv_file = open(output_filename, 'w', newline='')
        output_writer = csv.writer(csv_file)

        for start_time in arg_motion_start_times_list:
            start_time_str = str(start_time)
            output_writer.writerow([start_time_str])

        logger.debug("Finished writing motion file: " + output_filename)
        return