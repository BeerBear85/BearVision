
from scipy.signal import butter, filtfilt


class CameraViewGenerator:
    def __init__(self):
        # bbox_parameters
        self.box_width_scale = 8
        self.box_aspect_ratio = 16/9 #Standard aspect ratio that is fairly wide
        self.pos_filter_cutoff = 0.4  # Desired cutoff frequency of the filter, Hz
        self.fps = None  # Sample rate, Hz
        self.frame_width = None
        self.frame_height = None
        

    def init(self, video_fps, frame_width, frame_height):
        self.fps = video_fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        return
    
    def calculate_camera_view_size(self, box_log):
        
        #find the average size of the bounding boxes
        box_width = [inner_list[2] for inner_list in box_log if inner_list]
        box_height = [inner_list[3] for inner_list in box_log if inner_list]
        avg_box_width = int(sum(box_width)/len(box_width))
        avg_box_height = int(sum(box_height)/len(box_height))
        print(f'Average box width: {avg_box_width} Average box height: {avg_box_height}')
        camera_view_width = int(avg_box_width * self.box_width_scale)
        camera_view_height = int(camera_view_width / self.box_aspect_ratio)
        print(f'Camera width: {camera_view_width} Camera height: {camera_view_height}')
        return camera_view_width, camera_view_height

    def filter_position(self, state_log):
        ## Filter the measured position for a smoother camera path
        x = [inner_list[0] for inner_list in state_log if inner_list]
        y = [inner_list[1] for inner_list in state_log if inner_list]

        #Filter parameters
        order = 1
        nyq = 0.5 * self.fps
        normal_cutoff = self.pos_filter_cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        x_smooth = filtfilt(b, a, x)
        y_smooth = filtfilt(b, a, y)

        if False:
            # Plot the filtered data
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(x, y, label='Original Data')
            plt.plot(x_smooth, y_smooth, label='Filtered Data', linewidth=2, color='red', linestyle='--')
            #set limits to match the frame size
            plt.xlim(0, self.frame_width)
            plt.ylim(0, self.frame_height)
            plt.xlabel('x-position')
            plt.ylabel('y-position')
            plt.legend()
            plt.show()

        return x_smooth, y_smooth
    
    def calculate_view(self, x_pos_input, y_pos_input, camera_view_width, camera_view_height):
        # make list of bounding boxes with the new positions and sizes
        camera_view = []
        for i in range(len(x_pos_input)):
            x_pos = int(x_pos_input[i] - 0.5 * camera_view_width)
            y_pos = int(y_pos_input[i] - 0.5 * camera_view_height)
            if x_pos < 0:
                x_pos = 0
            elif x_pos + camera_view_width > self.frame_width:
                x_pos = self.frame_width - camera_view_width
            if y_pos < 0:
                y_pos = 0
            elif y_pos + camera_view_height > self.frame_height:
                y_pos = self.frame_height - camera_view_height
            camera_view.append([x_pos, y_pos, camera_view_width, camera_view_height])
        return camera_view


    def calculate(self, box_log, state_log):
        camera_view_width, camera_view_height = self.calculate_camera_view_size(box_log)
        x_pos, y_pos = self.filter_position(state_log)
        camera_view = self.calculate_view(x_pos, y_pos, camera_view_width, camera_view_height)
        return camera_view