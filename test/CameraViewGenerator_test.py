import logging


if __name__ == "__main__":
    write_mode = 'w'
else:
    write_mode = 'a'

logging.basicConfig(filename='debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)-8s:%(name)s:%(message)s',
                    filemode=write_mode)


if __name__ == "__main__":
    import sys
    import os
    import pickle
    from pprint import pprint
    import cv2

    modules_abs_path = os.path.abspath(os.path.join("code", "modules"))
    sys.path.append(modules_abs_path)
    from CameraViewGenerator import CameraViewGenerator

    logger = logging.getLogger(__name__)


    input_video = os.path.abspath(os.path.join("test", "test_video", "TestMovie1.mp4"))
    #input_video = os.path.abspath(os.path.join("test", "test_video", "TestMovie2.mp4"))
    #input_video = os.path.abspath(os.path.join("test", "test_video", "TestMovie3.avi"))
    #input_video = os.path.abspath(os.path.join("test", "test_video", "TestMovie4.avi"))

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

    # Run the actual module that is being tested
    my_camera_view_generator = CameraViewGenerator()
    my_camera_view_generator.init(data['fps'], frame_width, frame_height)
    camera_view = my_camera_view_generator.calculate(box_log, state_log)


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




    
