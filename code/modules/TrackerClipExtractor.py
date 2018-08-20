import cv2, time, os, datetime, logging
from multiprocessing.pool import Pool
from functools import partial

import BasicClipSpecification
import GoproVideo
import BearTracker
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)  #Set logger to reflect the current file


class TrackerClipExtractor:
    def __init__(self):
        logger.debug("TrackerClipExtractor created")
        self.tmp_options = ConfigurationHandler.get_configuration()
        self.number_of_process_workers = int(self.tmp_options['TRACKER_CLIP_SPECIFICATION']['number_of_process_workers'])

        return

    def extract_clips_from_list(self, arg_clip_specification_list: list):
        logger.info("Extracting tracker clips from specifications")

        # Single process
        for clip_spec in arg_clip_specification_list:
            TrackerClipExtractor.extract_single_clip(self.tmp_options, clip_spec)

        # multiprocess
        #with Pool(processes=self.number_of_process_workers) as pool:
        #    pool.map(partial(TrackerClipExtractor.extract_single_clip, self.tmp_options), arg_clip_specification_list)

        return

    @staticmethod
    def extract_single_clip(arg_option_obj: ConfigurationHandler, arg_clip_spec: BasicClipSpecification):
        logger.info("Creating output file: " + arg_clip_spec.output_video_path)
        tmp_input_video = GoproVideo.GoproVideo(arg_option_obj)
        tmp_input_video.init(arg_clip_spec.video_file_path)

        tmp_tracker = BearTracker.BearTracker(tmp_input_video.fps)

        # Parameter reads
        tmp_max_duration = datetime.timedelta(seconds=float(arg_option_obj['TRACKER_CLIP_SPECIFICATION']['max_clip_duration']))
        tmp_output_video_relative_speed = float(arg_option_obj['TRACKER_CLIP_SPECIFICATION']['output_video_speed'])
        tmp_show_tracker_debug = bool(arg_option_obj['TRACKER_CLIP_SPECIFICATION']['show_tracker_debug'])

        # Derived parameters
        tmp_start_frame = int(tmp_input_video.fps * (arg_clip_spec.start_time - tmp_input_video.creation_time).total_seconds())
        tmp_output_fps = int(tmp_input_video.fps * tmp_output_video_relative_speed)
        tmp_clip_frame_max_duration = int(tmp_input_video.fps * tmp_max_duration.total_seconds())

        # Init tracker

        # Start reading video file
        for iterator in range(0, tmp_clip_frame_max_duration):
            # print("Extracting frame: " + str(relative_frame_number))
            read_return_value, frame, abs_frame_number = tmp_input_video.read_frame()
            if read_return_value == 0:  # end of file
                break
            if read_return_value == 20:  # GoPro video error
                continue

        # Iterate tracker


        # Debug draw update (if debug enabled - see motion start detector)


        # Post process (rate limit on decline speed)? - for later.

        # Write cut clip (own function)
        #write_clip(tmp_input_video)



    @staticmethod
    def write_clip(arg_gopro_video: GoproVideo):
        #tmp_input_video.set_start_point(start_frame)  # Set start point of video
        output_codex = cv2.VideoWriter_fourcc(*'DIVX')
        #writer_object = cv2.VideoWriter(arg_clip_spec.output_video_path, output_codex, output_fps, (clip_frame_width, clip_frame_height))

        #writer_object.write(frame)

        #writer_object.release()