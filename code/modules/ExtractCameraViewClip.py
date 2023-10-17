import os
import pickle
import cv2

from CameraViewGenerator import CameraViewGenerator

class ExtractCameraViewClip:
    def __init__(self):
        self.my_camera_view_generator = CameraViewGenerator()
        self.tracker_data = None

        # Output video parameters
        self.clip_fps = 30
        self.clip_frame_height = 720 #720p standard resolution
        self.aspect_ratio = 16/9
        self.clip_frame_width = int(self.clip_frame_height * self.aspect_ratio)
        return
        

    def init(self, pickle_file_name):
        with open(pickle_file_name, "rb") as f:
            self.tracker_data = pickle.load(f)
        f.close()
        return

    
    def run(self):
        # Get the data from the pickle file
        state_log = self.tracker_data['state_log']
        box_log = self.tracker_data['box_log']
        start_frame = self.tracker_data['start_frame']
        video_file_name = self.tracker_data['video_file_name']

        # Open the video file
        cap = cv2.VideoCapture(video_file_name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.my_camera_view_generator.init(fps, frame_width, frame_height)
        camera_view = self.my_camera_view_generator.calculate(box_log, state_log)

        # Initialize the video capture
        output_video_path = os.path.abspath(f'{video_file_name}_camera_view.avi')
        output_codex = cv2.VideoWriter_fourcc(*'DIVX')
        writer_object = cv2.VideoWriter(output_video_path, output_codex, self.clip_fps,
                                        (self.clip_frame_width, self.clip_frame_height))

        current_frame_number = 0
        camera_view_index = 0

        # Loop through each frame of the video
        while True:
            # Read the frame
            ret, frame = cap.read()

            # If there are no more frames, break out of the loop
            if not ret:
                break

            current_frame_number += 1
            if (current_frame_number >= start_frame) and (camera_view_index < len(camera_view)):
                # Get the bounding box coordinates for the current frame
                bbox = camera_view[camera_view_index]
                camera_view_index += 1

                # Extract the camera view from the frame
                camera_view_frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                #Resize the camera view to the desired output size
                camera_view_frame = cv2.resize(camera_view_frame, (self.clip_frame_width, self.clip_frame_height), 0, 0, interpolation=cv2.INTER_LINEAR)
                writer_object.write(camera_view_frame)

        # Release the video objects
        cap.release()
        writer_object.release()

        return