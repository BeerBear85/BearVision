# Make main process create the new file and sub-processes append - probably not the nices way of doing this
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
    import matplotlib.pyplot as plt
    import cv2
    import pickle

    modules_abs_path = os.path.abspath("code/modules")
    dnn_models_abs_path = os.path.abspath("code/dnn_models")

    sys.path.append(modules_abs_path)
    sys.path.append(dnn_models_abs_path)

    from BearTracker import BearTracker
    from BearTracker import State
    from DnnHandler import DnnHandler



    logger = logging.getLogger(__name__)

    #input_video = os.path.abspath("test/test_video/TestMovie1.mp4")
    #input_video = os.path.abspath("test/test_video/TestMovie2.mp4")
    input_video = os.path.abspath("test/test_video/TestMovie3.avi")
    #input_video = os.path.abspath("test/test_video/TestMovie4.avi")

    tracker = BearTracker()
    
    #Read frames from video
    #Check if file exists
    if not os.path.isfile(input_video):
        print(f'Could not find file {input_video}')
        sys.exit(1)
    cap = cv2.VideoCapture(input_video)
    #get fps of video
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    start_frame = 0

    tracker.init()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video')
            break
        frame_count += 1
        start_state = tracker.state

        if tracker.state == State.SEARCHING:
            start_frame = frame_count #stops updating start_frame after first frame

        tracker.calculate(frame)
        if start_state == State.TRACKING:
            frame = tracker.draw(frame)

        #Scale frame to 50% for better overview
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('frame', frame)
        #Wait for 1 ms for keypress
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if tracker.state == State.DONE:
            print('Bear went out of frame')
            break

    cap.release()
    # take x-position for logged state

    x_pos_log = [inner_list[0] for inner_list in tracker.state_log if inner_list]
    y_pos_log = [inner_list[1] for inner_list in tracker.state_log if inner_list]

    #plot x and y position
    plt.figure()
    plt.plot(x_pos_log, y_pos_log)
    #set axis limits to match the frame size
    plt.xlim([0, frame_width])
    plt.ylim([0, frame_height])
    plt.show()

    # save the etire state log
    pickle_file_name = input_video.split('.')[0] + '_tracking_vars.pkl'

    data = {
        'state_log': tracker.state_log,
        'box_log': tracker.box_log,
        'start_frame': start_frame,
        'fps': fps,
        'frame_width': int(frame.shape[1]),
        'frame_height': int(frame.shape[0]),
    }

    with open(pickle_file_name, 'wb') as f:
        pickle.dump(data, f)






