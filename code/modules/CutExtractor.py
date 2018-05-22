import cv2, time, os, datetime, logging

import FullClipSpecification
import GoproVideo

logger = logging.getLogger(__name__)  #Set logger to reflect the current file

output_codex = cv2.VideoWriter_fourcc(*'DIVX')
output_video_speed = 0.5  # 0.5 means that a video clip output should be half speed of input file


class CutExtractor:
    def __init__(self):
        self.input_video = GoproVideo.GoproVideo()  # Make one GoPro object which is reinitilised if input video changes
        return

    def extract_full_clip_specifications(self, arg_clip_specification_list: list):
        for clip_spec in arg_clip_specification_list:
            logger.info("Creating output file: " + clip_spec.output_video_path)
            self.input_video.init(clip_spec.video_file.path)

            start_frame = int(self.input_video.fps * (clip_spec.start_time - self.input_video.creation_time).total_seconds())
            output_fps = int(self.input_video.fps * output_video_speed)
            clip_frame_duration = int(self.input_video.fps * clip_spec.duration.total_seconds())

            self.input_video.set_start_point(start_frame)  # Set start point of video
            writer_object = cv2.VideoWriter(clip_spec.output_video_path, output_codex, output_fps, (self.input_video.width, self.input_video.height))

            ### Read frames and write cut_out ###
            for relative_frame_number in range(0, clip_frame_duration):
                # print("Extracting frame: " + str(relative_frame_number))
                read_return_value, frame, abs_frame_number = self.input_video.read_frame()
                if (read_return_value == 0):  # end of file
                    break
                writer_object.write(frame)

            writer_object.release()
        return