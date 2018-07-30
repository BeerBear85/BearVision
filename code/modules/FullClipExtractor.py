import cv2, time, os, datetime, logging
from multiprocessing.pool import Pool
from functools import partial

import FullClipSpecification
import GoproVideo
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)  #Set logger to reflect the current file


class FullClipExtractor:
    def __init__(self):
        logger.debug("FullClipExtractor created")
        self.tmp_options = ConfigurationHandler.get_configuration()
        self.number_of_process_workers = int(self.tmp_options['FULL_CLIP_SPECIFICATION']['number_of_process_workers'])
        return

    def extract_full_clip_specifications(self, arg_clip_specification_list: list):
        logger.info("Extracting full clips from specifications")

        # Single process
        #for clip_spec in arg_clip_specification_list:
        #    FullClipExtractor.extract_single_clip(self.tmp_options, clip_spec)

        # multiprocess
        with Pool(processes=self.number_of_process_workers) as pool:
            pool.map(partial(FullClipExtractor.extract_single_clip, self.tmp_options), arg_clip_specification_list)

        return

    @staticmethod
    def test_1(arg_clip_spec: FullClipSpecification):
        print(arg_clip_spec.output_video_scale)

    @staticmethod
    def extract_single_clip(arg_option_obj: ConfigurationHandler, arg_clip_spec: FullClipSpecification):
        logger.info("Creating output file: " + arg_clip_spec.output_video_path)
        tmp_input_video = GoproVideo.GoproVideo(arg_option_obj)
        tmp_input_video.init(arg_clip_spec.video_file_path)

        start_frame = int(tmp_input_video.fps * (arg_clip_spec.start_time - tmp_input_video.creation_time).total_seconds())
        output_fps = int(tmp_input_video.fps * arg_clip_spec.output_video_relative_speed)
        clip_frame_duration = int(tmp_input_video.fps * arg_clip_spec.duration.total_seconds())
        clip_frame_width = int(tmp_input_video.width * arg_clip_spec.output_video_scale)
        clip_frame_height = int(tmp_input_video.height * arg_clip_spec.output_video_scale)

        tmp_input_video.set_start_point(start_frame)  # Set start point of video
        output_codex = cv2.VideoWriter_fourcc(*'DIVX')
        writer_object = cv2.VideoWriter(arg_clip_spec.output_video_path, output_codex, output_fps,
                                        (clip_frame_width, clip_frame_height))

        ### Read frames and write cut_out ###
        for iterator in range(0, clip_frame_duration):
            # print("Extracting frame: " + str(relative_frame_number))
            read_return_value, frame, abs_frame_number = tmp_input_video.read_frame()
            if read_return_value == 0:  # end of file
                break
            if read_return_value == 20:  # GoPro video error
                continue

            if arg_clip_spec.output_video_scale != 1.0:
                frame = cv2.resize(frame, (clip_frame_width, clip_frame_height), 0, 0, interpolation=cv2.INTER_LINEAR)

            writer_object.write(frame)

        writer_object.release()