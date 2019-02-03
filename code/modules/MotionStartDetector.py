
import logging, os, cv2, datetime, csv
import numpy as np
import GoproVideo
import ast
import pandas as pd
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

        tmp_show_video_debug = bool(int(arg_option_obj['MOTION_DETECTION']['show_video_debug']))
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
        logger.debug("Looking for motion in the following pixel range (y_start,y_end,x_start,x_end): " + str(tmp_absolute_search_box_dimensions))

        next_allowed_motion_frame = 0
        motion_frame_counter = 0
        motion_start_time_list = []
        motion_start_info_names = ['time', 'bbox']
        motion_start_info = pd.DataFrame(columns=motion_start_info_names)

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

                    # Find motion time
                    relative_start_time_double = (frame_number - tmp_motion_frame_counter_threshold) / MyGoproVideo.fps # [s]
                    relative_start_time = datetime.timedelta(seconds=int(relative_start_time_double), milliseconds=int((relative_start_time_double-int(relative_start_time_double))*1000))
                    abs_motion_start_time = MyGoproVideo.creation_time + relative_start_time
                    logger.debug("Motion detected at frame: " + str(frame_number) + ", corrisponding to relative time: " + str(relative_start_time) + ", absolute time: " + abs_motion_start_time.strftime("%Y%m%d_%H_%M_%S_%f"))
                    print("Motion detected at frame: " + str(frame_number) + ", corrisponding to relative time: " + str(relative_start_time) + ", absolute time: " + abs_motion_start_time.strftime("%Y%m%d_%H_%M_%S_%f"))
                    tmp_bbox = MotionStartDetector.get_bounding_box(mask,(MyGoproVideo.width, MyGoproVideo.height), tmp_absolute_search_box_dimensions)

                    motion_start_time_list.append(abs_motion_start_time)
                    motion_start_info = motion_start_info.append(pd.DataFrame([[abs_motion_start_time, tmp_bbox]], columns=motion_start_info_names), ignore_index=True)

                    # Draw local bounding box
                    tmp_p1 = (tmp_bbox[0], tmp_bbox[1])
                    tmp_p2 = (tmp_bbox[0] + tmp_bbox[2],
                              tmp_bbox[1] + tmp_bbox[3])
                    cv2.rectangle(frame, tmp_p1, tmp_p2, (255, 0, 0), 2)
                    frame = cv2.resize(frame, (int(0.4*MyGoproVideo.width), int(0.4*MyGoproVideo.height)), 0, 0, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('frame with bbox', frame)
                    cv2.waitKey(1)


            if tmp_show_video_debug:
                #cv2.imshow('frame', frame)
                cv2.imshow('frame_cut', frame_cut)
                cv2.imshow('mask', mask)
                cv2.waitKey(1)

            #if len(motion_start_info) > 1:
            #    return motion_start_info

        logger.info("Finished detection motion in file: " + arg_video_for_process_path)
        return motion_start_info

    @staticmethod
    def process_video(arg_option_obj, arg_video_for_process_path):
        motion_start_info = MotionStartDetector.find_motion_start_times(arg_option_obj, arg_video_for_process_path)
        MotionFilesHandler.write_motion_file(arg_option_obj, arg_video_for_process_path, motion_start_info)
        return

    @staticmethod
    def get_bounding_box(arg_mask, arg_frame_dimension, arg_cut_dimensions):
        # Setup SimpleBlobDetector parameters.
        tmp_blob_params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        tmp_blob_params.minThreshold = 1
        tmp_blob_params.maxThreshold = 200

        # Filter by Area.
        tmp_blob_params.filterByArea = False
        tmp_blob_params.minArea = 1500
        tmp_blob_params.maxArea = 100000

        # Filter by Circularity
        tmp_blob_params.filterByCircularity = False
        tmp_blob_params.minCircularity = 0.1

        # Filter by Convexity
        tmp_blob_params.filterByConvexity = False
        tmp_blob_params.minConvexity = 0.1

        # Filter by Inertia
        tmp_blob_params.filterByInertia = False
        tmp_blob_params.minInertiaRatio = 0.01

        tmp_blob_params.filterByColor = False

        # Set up the blob detector with parameters.
        tmp_detector = cv2.SimpleBlobDetector_create(tmp_blob_params)

        # Make som image morphological operations such as Dilation and cv2.MORPH_CLOSE
        # See https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html

        tmp_morph_size = int(10)
        morph_kernel = np.ones((tmp_morph_size, tmp_morph_size), np.uint8)
        #cv2.imshow('Raw mask', arg_mask)
        arg_mask = cv2.dilate(arg_mask,morph_kernel,iterations = 1)
        #cv2.imshow('Dilated mask', arg_mask)
        arg_mask = cv2.morphologyEx(arg_mask, cv2.MORPH_CLOSE, morph_kernel)
        #cv2.imshow('Closed mask', arg_mask)

        # Detect blobs.
        tmp_keypoints = tmp_detector.detect(arg_mask)

        # Find keypoint with largest radius
        largest_keypoint = None
        for keypoint in tmp_keypoints:
            #print("Size: %d", keypoint.size)
            #print("X,Y: %d, %d", keypoint.pt[0], keypoint.pt[1])
            if largest_keypoint is None:
                largest_keypoint = keypoint
            elif keypoint.size > largest_keypoint.size:
                largest_keypoint = keypoint

        # Handle the no keypoint case with a warning
        if largest_keypoint is None:
            logger.warning("No blobs found in mask!")

        # Create bounding box from keypoint
        # Remember that the mask is only a sub part of the full video frame
        # x1,y1,width,height
        tmp_local_bounding_box = [
            int(largest_keypoint.pt[0] - largest_keypoint.size/2),
            int(largest_keypoint.pt[1] - largest_keypoint.size/2),
            int(largest_keypoint.size),
            int(largest_keypoint.size)
        ]

        # You need the full image size info and the cut location info to get real location and do range checking!
        # x1,y1,width,height
        tmp_full_bounding_box = [
            tmp_local_bounding_box[0] + arg_cut_dimensions[2],
            tmp_local_bounding_box[1] + arg_cut_dimensions[0],
            tmp_local_bounding_box[2],
            tmp_local_bounding_box[3]
        ]

        # Scale the bounding box around the center
        # TODO make configfile parameter!
        tmp_scaling_factor = float(3)

        # Dummy: Check that it works for scale factor of 1!!!
        tmp_box_dim_change = (tmp_full_bounding_box[2] * (tmp_scaling_factor - 1),
                              tmp_full_bounding_box[3] * (tmp_scaling_factor - 1))

        tmp_full_bounding_box = [
            int(tmp_full_bounding_box[0] - tmp_box_dim_change[0]/2.0),
            int(tmp_full_bounding_box[1] - tmp_box_dim_change[1]/2.0),
            int(tmp_full_bounding_box[2] * tmp_scaling_factor),
            int(tmp_full_bounding_box[3] * tmp_scaling_factor)
        ]

        # Range check
        if tmp_full_bounding_box[0] < 0:
            tmp_full_bounding_box[0] = int(0)

        if tmp_full_bounding_box[1] < 0:
            tmp_full_bounding_box[1] = int(0)

        tmp_x_end = tmp_full_bounding_box[0] + tmp_full_bounding_box[2]
        if tmp_x_end >= arg_frame_dimension[0]: # width check
            tmp_x_offset =  (tmp_x_end - arg_frame_dimension[0]) + 1
            tmp_full_bounding_box[0] = tmp_full_bounding_box[0] - tmp_x_offset

        tmp_y_end = tmp_full_bounding_box[1] + tmp_full_bounding_box[3]
        if tmp_y_end >= arg_frame_dimension[0]: # height check
            tmp_y_offset = (tmp_y_end - arg_frame_dimension[1]) + 1
            tmp_full_bounding_box[1] = tmp_full_bounding_box[1] - tmp_y_offset

        logger.debug("Full bounding box spec: " + str(tmp_full_bounding_box))

        show_bbox_debug = False
        if show_bbox_debug:
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            tmp_mask_with_keypoints = cv2.drawKeypoints(arg_mask, tmp_keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Draw local bounding box
            tmp_p1 = (tmp_local_bounding_box[0], tmp_local_bounding_box[1])
            tmp_p2 = (tmp_local_bounding_box[0] + tmp_local_bounding_box[2], tmp_local_bounding_box[1] + tmp_local_bounding_box[3])
            cv2.rectangle(tmp_mask_with_keypoints, tmp_p1, tmp_p2, (255, 0, 0), 2)

            cv2.imshow('Final blob detection', tmp_mask_with_keypoints)
            cv2.waitKey(0)

        return tmp_full_bounding_box
