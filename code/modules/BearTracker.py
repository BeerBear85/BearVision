import cv2, math
from csv import writer
import numpy as np

from DnnHandler import DnnHandler

from scipy.signal import filtfilt, butter  # http://scipy-cookbook.readthedocs.io/items/FiltFilt.html

class BearTracker:
    def __init__(self):

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
        self.state_log = list()

        self.dnn_handler = DnnHandler()

    def init(self, arg_bbox):
        start_pos = (int(arg_bbox[0] + 0.5*arg_bbox[2]), int(arg_bbox[1] + 0.5*arg_bbox[3]))
        self.kalman_filter.statePost = np.array([[start_pos[0]], [start_pos[1]], [0], [0]], np.float32)
        self.dnn_handler.init()

        return

    def update(self, arg_frame):
        ## Prediction ##
        self.latest_state_estimate = self.kalman_filter.predict()  # Predicted state from motion model

        ## Measurement ##
        #TODO: only do this for a sub-part of the frame (covariance of the position)
        [boxes, confidences] = self.dnn_handler.find_person(arg_frame)
        if len(boxes) != 0:
            tmp_bbox = boxes[0] #just take the first one for now
            tmp_measurement = self.get_bbox_center(tmp_bbox)

            ## Correction ##
            self.latest_state_estimate = self.kalman_filter.correct(tmp_measurement)

        ## Logging ##
        self.log_state(self.latest_state_estimate)

        #Tell if the X position is outside the right side of the frame
        if int(self.latest_state_estimate[0]) > arg_frame.shape[1]:
            return True
        else:
            return False

    def draw(self, frame):
        tmp_color = (0, 0, 255) #red
        tmp_pos = (int(self.latest_state_estimate[0]), int(self.latest_state_estimate[1]))
        cv2.circle(frame, tmp_pos, 10, tmp_color, -1)
        cv2.putText(frame, "X-velocity : " + str(self.latest_state_estimate[2]), (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 170, 50), 2)

        tmp_x_sigma = math.sqrt(self.kalman_filter.errorCovPost[0, 0])
        tmp_y_sigma = math.sqrt(self.kalman_filter.errorCovPost[1, 1])

        # print("errorCovPost - type: " + str(type(self.kalman_filter.errorCovPost)) + " value: " + str(self.kalman_filter.errorCovPost))
        #print("tmp_x_sigma - type: " + str(type(tmp_x_sigma)) + " value: " + str(tmp_x_sigma))

        cv2.ellipse(frame, tmp_pos, (int(tmp_x_sigma*3), int(tmp_y_sigma*3)), 0, 0, 360, tmp_color, 2)
        return frame

    def get_bbox_center(self, bbox):
        tmp_x = int(bbox[0] + 0.5*bbox[2])
        tmp_y = int(bbox[1] + 0.5*bbox[3])
        return np.array([[tmp_x], [tmp_y]], np.float32)

    def log_state(self, arg_state):
        tmp_state_vec = [int(arg_state[0]), int(arg_state[1]), float(arg_state[2]), float(arg_state[3])]
        self.state_log.append(tmp_state_vec)
        print(tmp_state_vec)
        return

    def write_state_log_file(self):
        output_filename = "state_log.csv"
        #csv_file = open(output_filename, 'w', newline='')
        #output_writer = writer(csv_file)

        #filter test
        np_log = np.matrix(self.state_log)
        # Create an order 3 lowpass butterworth filter.
        b, a = butter(3, 0.05)
        pos_x = filtfilt(b, a, np_log[:, 0].T)
        pos_y = -filtfilt(b, a, np_log[:, 1].T)
        vel_x = filtfilt(b, a, np_log[:, 2].T)
        vel_y = -filtfilt(b, a, np_log[:, 3].T)
        #print(pos_x.T)
        #print(type(pos_x))
        np_log_filteren = np.concatenate([pos_x, pos_y, vel_x, vel_y]).T
        #print(np_log_filteren)

        #for state in self.state_log:
        #    output_writer.writerow(np_log_filteren)

        np.savetxt(output_filename, np_log_filteren, delimiter=",")

