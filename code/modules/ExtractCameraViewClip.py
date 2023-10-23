import os
import pickle
import cv2

from CameraViewGenerator import CameraViewGenerator
from GoproVideo import GoproVideo

class ExtractCameraViewClip:
    def __init__(self):
        self.my_camera_view_generator = CameraViewGenerator()
        self.input_video_obj = GoproVideo()

        self.tracker_data = None

        # Output video parameters
        self.clip_fps = 30
        self.clip_frame_height = 720 #720p standard resolution
        self.aspect_ratio = 16/9
        self.clip_frame_width = int(self.clip_frame_height * self.aspect_ratio)
        self.output_codex = cv2.VideoWriter_fourcc(*'DIVX')
        return

    def init(self):
        return

    def run(self, pickle_file_name):
        print(f'Starting extraction for {pickle_file_name}')
        with open(pickle_file_name, "rb") as f:
            self.tracker_data = pickle.load(f)
            f.close()

        # Get the data from the pickle file
        state_log = self.tracker_data['state_log']
        box_log = self.tracker_data['box_log']
        start_frame = self.tracker_data['start_frame']
        video_file_name = self.tracker_data['video_file_name']

        # Open the video file
        self.input_video_obj.init(video_file_name)
        frame_width = self.input_video_obj.width
        frame_height = self.input_video_obj.height
        fps = self.input_video_obj.fps

        # Generate camera view
        self.my_camera_view_generator.init(fps, frame_width, frame_height)
        camera_view = self.my_camera_view_generator.calculate(box_log, state_log)

        # Initialize the video writer
        output_video_file_name = os.path.splitext(video_file_name)[0] # remove the file extension from the video file name
        output_video_path = os.path.abspath(f'{output_video_file_name}_{start_frame}_camera_view.avi')

        writer_object = cv2.VideoWriter(output_video_path, self.output_codex, self.clip_fps,
                                        (self.clip_frame_width, self.clip_frame_height))

        self.input_video_obj.set_start_point(start_frame)
        camera_view_index = 0
        camera_view_end_index = start_frame + len(camera_view)

        # Loop through each frame of the video
        while True:
            # Read the frame
            read_return_value, frame, frame_number = self.input_video_obj.read_frame()
            print(f'Frame number: {frame_number} - stop at {camera_view_end_index}')
            if read_return_value == 0:
                print('Reached end of video')
                break
            if read_return_value == 20: #GoPro error with empty frame
                continue

            if (frame_number < camera_view_end_index):
                # Get the bounding box coordinates for the current frame
                bbox = camera_view[camera_view_index]
                camera_view_index += 1

                # Extract the camera view from the frame
                camera_view_frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                #Resize the camera view to the desired output size
                camera_view_frame = cv2.resize(camera_view_frame, (self.clip_frame_width, self.clip_frame_height), 0, 0, interpolation=cv2.INTER_LINEAR)
                writer_object.write(camera_view_frame)
            else:
                print('Reached end of camera view list')
                break

        # Release the video objects
        writer_object.release()
        print(f'Extract complete for {pickle_file_name}')
        return


    def extract_folder(self, folder_path):
        #Create list of all pickle files in folder
        pickle_file_name_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
        
        pickle_file_name_list.sort() #sort the list

        for pickle_file_name in pickle_file_name_list:
            self.run(pickle_file_name)
            
        return
