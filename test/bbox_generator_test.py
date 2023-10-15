import logging
import numpy as np
from scipy.signal import butter, filtfilt

if __name__ == "__main__":
    write_mode = 'w'
else:
    write_mode = 'a'

logging.basicConfig(filename='debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)-8s:%(name)s:%(message)s',
                    filemode=write_mode)

def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as plt
    import pickle
    from pprint import pprint
    import cv2

    modules_abs_path = os.path.abspath("code/modules")

    logger = logging.getLogger(__name__)

    # bbox_parameters
    box_width_scale = 8
    box_aspect_ratio = 16/9

    input_video = os.path.abspath("test/test_video/TestMovie1.mp4")
    #input_video = os.path.abspath("test/test_video/TestMovie2.mp4")
    #input_video = os.path.abspath("test/test_video/TestMovie3.avi")
    #input_video = os.path.abspath("test/test_video/TestMovie4.avi")

    pickle_file_name = input_video.split('.')[0] + '_tracking_vars.pkl'
    
    with open(pickle_file_name, "rb") as f:
        data = pickle.load(f)
    f.close()
    # show what data is in the pickle file
    pprint(data.keys())

    state_log = data['state_log']
    box_log = data['box_log']
    start_frame = data['start_frame']
    frame_width = data['frame_width']
    frame_height = data['frame_height']

    #find the average size of the bounding boxes
    box_width = [inner_list[2] for inner_list in box_log if inner_list]
    box_height = [inner_list[3] for inner_list in box_log if inner_list]
    avg_box_width = int(sum(box_width)/len(box_width))
    avg_box_height = int(sum(box_height)/len(box_height))
    print(f'Average box width: {avg_box_width} Average box height: {avg_box_height}')
    camera_view_width = int(avg_box_width * box_width_scale)
    camera_view_height = int(camera_view_width / box_aspect_ratio)
    print(f'Camera width: {camera_view_width} Camera height: {camera_view_height}')

    ## Filter the measured position for a smoother camera path
    x = [inner_list[0] for inner_list in state_log if inner_list]
    y = [inner_list[1] for inner_list in state_log if inner_list]

    #Filter parameters
    cutoff = 2  # desired cutoff frequency of the filter, Hz
    fs = data['fps']  # sample rate, Hz
    b, a = butter_lowpass(cutoff, fs)
    x_smooth = filtfilt(b, a, x)
    y_smooth = filtfilt(b, a, y)

    plt.figure()
    plt.plot(x, y, label='Original Data')
    plt.plot(x_smooth, y_smooth, label='Filtered Data', linewidth=2, color='red', linestyle='--')
    plt.legend()
    plt.show()

    # make list of bounding boxes with the new positions and sizes
    camera_view = []
    for i in range(len(x)):
        x_pos = int(x_smooth[i] - 0.5 * camera_view_width)
        y_pos = int(y_smooth[i] - 0.5 * camera_view_height)
        if x_pos < 0:
            x_pos = 0
        elif x_pos + camera_view_width > frame_width:
            x_pos = frame_width - camera_view_width
        if y_pos < 0:
            y_pos = 0
        elif y_pos + camera_view_height > frame_height:
            y_pos = frame_height - camera_view_height
        camera_view.append([x_pos, y_pos, camera_view_width, camera_view_height])
        


    # Open the video file
    cap = cv2.VideoCapture(input_video)

    # Create a window to display the video
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

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

        if current_frame_number >= start_frame:
            # Get the bounding box coordinates for the current frame
            bbox = camera_view[camera_view_index]
            camera_view_index += 1

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        # Display the frame in the window
        cv2.imshow('Video', frame)

        # Wait for a key press
        key = cv2.waitKey(50) & 0xFF

        # If the key pressed is 'q', break out of the loop
        if key == ord('q'):
            break

    # Release the video capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()




    
