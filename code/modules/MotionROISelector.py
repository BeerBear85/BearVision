
import logging, os, cv2, datetime, csv
import numpy as np
import GoproVideo
import ast
import pandas as pd
from tkinter import filedialog
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)

class MotionROISelector:
    def __init__(self):
        logger.debug("MotionROISelector created")
        tmp_options = ConfigurationHandler.get_configuration()

    def SelectROI(self, arg_input_video_folder):
        logger.debug("SelectROI() called")
        tmp_options = ConfigurationHandler.get_configuration()
        tmpVideoFileName = filedialog.askopenfilename(initialdir = arg_input_video_folder, title = "Select video file")

        MyGoproVideo = GoproVideo.GoproVideo(tmp_options)
        MyGoproVideo.init(tmpVideoFileName)
        MyGoproVideo.set_start_point(int(MyGoproVideo.frames/2)) # Start in the middle of the clip
        read_return_value, frame, frame_number = MyGoproVideo.read_frame()

        tmp_relative_search_box_dimensions = ast.literal_eval(tmp_options['MOTION_DETECTION']['search_box_dimensions'])  # [-] current ROI
        logger.debug("Current relative ROI: " + str(tmp_relative_search_box_dimensions))
        tmp_absolute_search_box_dimensions = [
            int(tmp_relative_search_box_dimensions[0] * MyGoproVideo.height), # Y start
            int(tmp_relative_search_box_dimensions[1] * MyGoproVideo.height), # Y end
            int(tmp_relative_search_box_dimensions[2] * MyGoproVideo.width),  # X start
            int(tmp_relative_search_box_dimensions[3] * MyGoproVideo.width)   # X end
        ]
        logger.debug("Current absolute ROI: " + str(tmp_absolute_search_box_dimensions))

        # Draw the current ROI
        tmp_p1 = (tmp_absolute_search_box_dimensions[0], tmp_absolute_search_box_dimensions[2])
        tmp_p2 = (tmp_absolute_search_box_dimensions[1], tmp_absolute_search_box_dimensions[3])
        cv2.rectangle(frame, tmp_p1, tmp_p2, (0, 0, 255), 4)


        AbsoluteROISelection = cv2.selectROI("Select the ROI for motion detection", frame, False)
        AbsoluteROISelection = (AbsoluteROISelection[0], AbsoluteROISelection[0]+AbsoluteROISelection[2], AbsoluteROISelection[1], AbsoluteROISelection[1]+AbsoluteROISelection[3]) # To match standart of BearVision (y_start,y_end,x_start,x_end)
        logger.debug("New absolute ROI: " + str(AbsoluteROISelection))

        RelativeROISelection = (AbsoluteROISelection[0] / MyGoproVideo.height,
                                AbsoluteROISelection[1] / MyGoproVideo.height,
                                AbsoluteROISelection[2] / MyGoproVideo.width,
                                AbsoluteROISelection[3] / MyGoproVideo.width,
                                )

        RelativeROISelection = [round(x, 2) for x in RelativeROISelection]
        logger.debug("New relative ROI: " + str(RelativeROISelection))

        tmp_options['MOTION_DETECTION']['search_box_dimensions'] = str(RelativeROISelection)

        return RelativeROISelection
