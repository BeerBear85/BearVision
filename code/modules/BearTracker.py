import cv2, math
import numpy as np

minimum_frames_between_resets = 30

class BearTracker:
    def __init__(self, arg_fps):
        self.internal_tracker = 0
        self.internal_tracker_bbox_size = (0, 0) # width, height

        model_position_noise_sigma = 1  # pixies
        model_velocity_noise_sigma = 1  # pixies
        measument_noise_sigma      = 10 # pixies
        self.kalman_filter = cv2.KalmanFilter(4, 2)  # States: x, y, v_x, v_y
        self.kalman_filter.transitionMatrix    = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # Const velocity model - should be TS (1/fps) in the position from velocity
        self.kalman_filter.measurementMatrix   = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) #Measurement: x, y
        self.kalman_filter.processNoiseCov     = np.diag(np.array([model_position_noise_sigma**2, model_position_noise_sigma**2, model_velocity_noise_sigma**2, model_velocity_noise_sigma**2 ], np.float32))
        self.kalman_filter.measurementNoiseCov = np.diag(np.array([measument_noise_sigma**2,measument_noise_sigma**2], np.float32))

        self.latest_state_estimate = 0
        self.kalman_filter.errorCovPost = np.diag(np.array([model_position_noise_sigma**2, model_position_noise_sigma**2, model_velocity_noise_sigma**2, model_velocity_noise_sigma**2 ], np.float32))
        self.kalman_filter.statePost    = np.array([[300], [800], [0], [0]], np.float32)
        self.max_velocity = 0
        self.frame_since_reset = int(0)

    def init(self, arg_frame, arg_bbox):
        self.internal_tracker_bbox_size = (arg_bbox[2], arg_bbox[3])
        # self.internal_tracker = cv2.TrackerMIL_create()
        self.internal_tracker = cv2.TrackerKCF_create()  # it is not enough to just run the init() again to reset
        return self.internal_tracker.init(arg_frame, arg_bbox)

    def update(self, arg_frame):
        [tmp_tracker_status, tmp_bbox] = self.internal_tracker.update(arg_frame)
        tmp_tracker_measurement = self.get_bbox_center(tmp_bbox)

        tmp_prediction = self.kalman_filter.predict()  # Predicted state from motion model

        # if ((tmp_tracker_measurement[0] + 10) < tmp_prediction[0]):
        #     print("X measurement lower than expected")
        #     self.latest_state_estimate = tmp_prediction
        # else:
        if tmp_tracker_status:
            self.latest_state_estimate = self.kalman_filter.correct(tmp_tracker_measurement)

        if (self.latest_state_estimate[2] > self.max_velocity) & (self.frame_since_reset > minimum_frames_between_resets): #right after a reset, the velocity goes faulsly high
            self.max_velocity = self.latest_state_estimate[2].copy()
            print("New max vel: " + str(self.max_velocity))

        self.frame_since_reset += int(1)
        if (self.latest_state_estimate[2] < 0) & (self.frame_since_reset > minimum_frames_between_resets):  # if x-velocity is negative, restart filter
            print("Restart of filter!!! - Low x-velocity")
            tmp_new_bbox = (tmp_bbox[0] + self.max_velocity*10, tmp_bbox[1], self.internal_tracker_bbox_size[0], self.internal_tracker_bbox_size[1]) #Todo: should reset on current estimate!
            self.init(arg_frame, tmp_new_bbox)
            self.frame_since_reset = int(0)

        if (not tmp_tracker_status) & (self.frame_since_reset > minimum_frames_between_resets):  # if x-velocity is negative, restart filter
            print("Restart of filter!!! - Tracking lost")
            tmp_new_bbox = (self.latest_state_estimate[0], self.latest_state_estimate[1], self.internal_tracker_bbox_size[0], self.internal_tracker_bbox_size[1])
            self.init(arg_frame, tmp_new_bbox)
            self.frame_since_reset = int(0)

        return (tmp_tracker_status, tmp_bbox)

    def draw(self, frame):
        tmp_color = (0, 0, 255) #red
        tmp_pos = (self.latest_state_estimate[0], self.latest_state_estimate[1])
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


    #kalman.correct(mp)
    #tp = kalman.predict()


    #Notes:
    #Hvis prediction er uden for bbox (tæt på), køres init af KCFen igen
    #Foretag ikke correct af Kalman filter når den målte x-position er X gange mindre end den predicted
