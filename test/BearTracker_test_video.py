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
    #input_video = os.path.abspath("test/test_video/TestMovie3.avi")
    input_video = os.path.abspath("test/test_video/TestMovie4.avi")


    #Read frames from video
    #Check if file exists
    if not os.path.isfile(input_video):
        print(f'Could not find file {input_video}')
        sys.exit(1)
    cap = cv2.VideoCapture(input_video)

    tracker = BearTracker()
    tracker.init(input_video)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video')
            break

        start_state = tracker.state

        tracker.calculate(frame)
        if start_state == State.TRACKING:
            frame = tracker.draw(frame)

        #Scale frame to 50% for better overview
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('frame', frame)
        #Wait for 1 ms for keypress
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if tracker.state == State.DONE:
            print('Bear went out of frame')
            break

    tracker.save_data()
    cap.release()


if False:
    x_pos_log = [inner_list[0] for inner_list in tracker.state_log if inner_list]
    y_pos_log = [inner_list[1] for inner_list in tracker.state_log if inner_list]

    #plot x and y position
    plt.figure()
    plt.plot(x_pos_log, y_pos_log)
    #set axis limits to match the frame size
    plt.show()


