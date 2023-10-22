
import math
import numpy as np
import cv2
import pickle
from enum import Enum

from DnnHandler import DnnHandler

class State(Enum):
    INIT = 1
    SEARCHING = 2
    TRACKING = 3
    DONE = 4

class BearTracker:
    def __init__(self):
        self.state = None
        self.search_window_width = 0.3
        self.search_window_height = 0.8

        model_position_noise_sigma = 10  # pixies
        model_velocity_noise_sigma = 1  # pixies
        measurement_noise_sigma      = 20 # pixies
        self.kalman_filter = cv2.KalmanFilter(4, 2)  # States: x, y, v_x, v_y
        self.kalman_filter.transitionMatrix    = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # Const velocity model - should be TS (1/fps) in the position from velocity
        self.kalman_filter.measurementMatrix   = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) #Measurement: x, y
        self.kalman_filter.processNoiseCov     = np.diag(np.array([model_position_noise_sigma**2, model_position_noise_sigma**2, model_velocity_noise_sigma**2, model_velocity_noise_sigma**2 ], np.float32))
        self.kalman_filter.measurementNoiseCov = np.diag(np.array([measurement_noise_sigma**2,measurement_noise_sigma**2], np.float32))

        self.latest_state_estimate = None
        self.kalman_filter.errorCovPost = np.diag(np.array([model_position_noise_sigma**2, model_position_noise_sigma**2, model_velocity_noise_sigma**2, model_velocity_noise_sigma**2 ], np.float32))
        self.kalman_filter.statePost    = np.array([[0], [0], [0], [0]], np.float32)

        self.search_area_scale = 20 #How much bigger the search area is compared to the current estimate covariance
        self.dnn_handler = DnnHandler()


        #For logging data
        self.state_log = list()
        self.box_log = list()
        self.start_frame = None
        self.video_file_name = None
        

    def init(self, arg_video_file_name):
        self.change_state(State.INIT)
        self.video_file_name = arg_video_file_name
        self.dnn_handler.init()
        self.start_frame = 0
        return
    
    def change_state(self, new_state):
        print(f'Changing state from {self.state} to {new_state}')
        self.state = new_state
        return
    
    def reset(self):
        self.change_state(State.INIT)
        return
    
    def calculate(self, arg_frame):
        if self.state == State.INIT:
            print('Tracker: Switching to SEARCHING state')
            self.change_state(State.SEARCHING)
            return
        elif self.state == State.SEARCHING:
            self.start_frame += 1 #stops updating start_frame after first frame
            found = self.search_for_start(arg_frame)
            if found:
                self.change_state(State.TRACKING)
            return
        elif self.state == State.TRACKING:
            still_in_picture = self.update(arg_frame)
            if not still_in_picture:
                self.change_state(State.DONE)
            return
        else:
            return False
    
    def search_for_start(self, arg_frame):
        #get frame size of video
        frame_width = int(arg_frame.shape[1])
        frame_height = int(arg_frame.shape[0])
        ROI_width = int(self.search_window_width * frame_width)
        ROI_height = int(self.search_window_height * frame_height)
        ROI_x = int(0)
        ROI_y = int(0.5 * frame_height - 0.5 * ROI_height)
        ROI_region = [ROI_x, ROI_y, ROI_width, ROI_height] #x, y, width, height
        frame_ROI = arg_frame[ROI_region[1]:ROI_region[1]+ROI_region[3], ROI_region[0]:ROI_region[0]+ROI_region[2]]
        [boxes, confidences] = self.dnn_handler.find_person(frame_ROI)
        if len(boxes) != 0:
            box_in_frame_ROI = boxes[0]
            box_in_frame = [box_in_frame_ROI[0] + ROI_x, box_in_frame_ROI[1] + ROI_y, box_in_frame_ROI[2], box_in_frame_ROI[3]]
            
            start_pos = (int(box_in_frame[0] + 0.5*box_in_frame[2]), int(box_in_frame[1] + 0.5*box_in_frame[3]))
            self.kalman_filter.statePost = np.array([[start_pos[0]], [start_pos[1]], [0], [0]], np.float32)
            return True
            
        return False

    
    def update(self, arg_frame):
        """Update the state estimate of the Kalman filter based on a new frame.
        Args:
            arg_frame: The input image frame as a NumPy array.
        Returns:
            bool: False if the position estimate is outside the frame - True otherwise.
        """
        ## Prediction ##
        self.latest_state_estimate = self.kalman_filter.predict()  # Predicted state from motion model

        ## Measurement ##
        search_box, search_frame = self.get_current_search_frame(arg_frame)

        [boxes, confidences] = self.dnn_handler.find_person(search_frame)
        # Map the box to the original frame
        boxes = [[box[0] + search_box[0], box[1] + search_box[1], box[2], box[3]] for box in boxes]

        if len(boxes) != 0:
            tmp_bbox = boxes[0] #just take the first one for now
            self.box_log.append(tmp_bbox)
            tmp_measurement = self.get_bbox_center(tmp_bbox)
            print(f'Person found at x: {tmp_measurement[0]}, y: {tmp_measurement[1]}')

            ## Correction ##
            self.latest_state_estimate = self.kalman_filter.correct(tmp_measurement)
        else:
            self.box_log.append(None) #to still have the same number of elements in the list

        ## Logging ##
        self.log_state(self.latest_state_estimate)

        #Tell if the X position is outside the frame
        if (int(self.latest_state_estimate[0]) < 0) or (int(self.latest_state_estimate[0]) > arg_frame.shape[1]):
            return False
        
        return True
    
    def get_current_search_frame(self, frame):
        search_box = self.get_current_search_box(frame)
        search_frame = frame[search_box[1]:search_box[1]+search_box[3], search_box[0]:search_box[0]+search_box[2]]

        return search_box, search_frame

    def get_current_search_box(self, frame):
        tmp_x_sigma = math.sqrt(self.kalman_filter.errorCovPost[0, 0])
        tmp_y_sigma = math.sqrt(self.kalman_filter.errorCovPost[1, 1])

        search_area_width = int(self.search_area_scale * tmp_x_sigma)
        search_area_height = int(self.search_area_scale * tmp_y_sigma)
        search_area_x = int(self.latest_state_estimate[0] - 0.5 * search_area_width)
        search_area_y = int(self.latest_state_estimate[1] - 0.5 * search_area_height)
        search_box = [search_area_x, search_area_y, search_area_width, search_area_height]

        # Range check
        if search_box[0] < 0:
            search_box[0] = 0
        if search_box[1] < 0:
            search_box[1] = 0
        if search_box[0] + search_box[2] > frame.shape[1]:
            search_box[2] = frame.shape[1] - search_box[0]
        if search_box[1] + search_box[3] > frame.shape[0]:
            search_box[3] = frame.shape[0] - search_box[1]

        return search_box

    def visualize_state(self, frame):
        tmp_color = (0, 0, 255) #red
        tmp_pos = (int(self.latest_state_estimate[0]), int(self.latest_state_estimate[1]))
        cv2.circle(frame, tmp_pos, 10, tmp_color, -1)
        cv2.putText(frame, "X-velocity : " + str(self.latest_state_estimate[2]), (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 170, 50), 2)

        tmp_x_sigma = math.sqrt(self.kalman_filter.errorCovPost[0, 0])
        tmp_y_sigma = math.sqrt(self.kalman_filter.errorCovPost[1, 1])

        cv2.ellipse(frame, tmp_pos, (int(tmp_x_sigma*3), int(tmp_y_sigma*3)), 0, 0, 360, tmp_color, 2)

        search_box = self.get_current_search_box(frame)
        cv2.rectangle(frame, (search_box[0], search_box[1]), (search_box[0]+search_box[2], search_box[1]+search_box[3]), tmp_color, 2)

        return frame

    def get_bbox_center(self, bbox):
        tmp_x = int(bbox[0] + 0.5*bbox[2])
        tmp_y = int(bbox[1] + 0.5*bbox[3])
        return np.array([[tmp_x], [tmp_y]], np.float32)

    def log_state(self, arg_state):
        tmp_state_vec = [int(arg_state[0]), int(arg_state[1]), float(arg_state[2]), float(arg_state[3])]
        self.state_log.append(tmp_state_vec)
        print(f'X: {tmp_state_vec[0]}, Y: {tmp_state_vec[1]}, X-velocity: {tmp_state_vec[2]}, Y-velocity: {tmp_state_vec[3]}')
        return

    def save_data(self):
        base_video_file_name = self.video_file_name.split('.')[0]
        file_name = f'{base_video_file_name}_{self.start_frame}_tracking.pkl'
        tmp_data = {
            'state_log': self.state_log,
            'box_log': self.box_log,
            'start_frame': self.start_frame,
            'video_file_name': self.video_file_name,
        }
        pickle.dump(tmp_data, open(file_name, 'wb'))
        return file_name

