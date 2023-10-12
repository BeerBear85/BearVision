# Make main process create the new file and sub-processes append - probably not the nices way of doing this
import logging
import cv2

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

    modules_abs_path = os.path.abspath("code/modules")
    dnn_models_abs_path = os.path.abspath("code/dnn_models")

    sys.path.append(modules_abs_path)
    sys.path.append(dnn_models_abs_path)

    from BearTracker import BearTracker
    from DnnHandler import DnnHandler



    logger = logging.getLogger(__name__)

    #input_video = os.path.abspath("test/test_video/TestMovie1.mp4")
    #input_video = os.path.abspath("test/test_video/TestMovie2.mp4")
    input_video = os.path.abspath("test/test_video/TestMovie3.avi")
    #input_video = os.path.abspath("test/test_video/TestMovie4.avi")

    tracker = BearTracker()
    dnn_handler = DnnHandler()
    dnn_handler.init()
    
    #Read frames from video
    #Check if file exists
    if not os.path.isfile(input_video):
        print(f'Could not find file {input_video}')
        sys.exit(1)
    cap = cv2.VideoCapture(input_video)
    #get fps of video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #get frame size of video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ROI_width = int(0.3 * frame_width)
    ROI_height = int(0.8 * frame_height)
    ROI_x = int(0)
    ROI_y = int(0.5 * frame_height - 0.5 * ROI_height)
    initial_ROI = [ROI_x, ROI_y, ROI_width, ROI_height] #x, y, width, height
    
    start_tracker = False
    frame_count = 0
    out_of_frame = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video')
            break
        frame_count += 1
        original_frame = frame
        #only take the part of the frame in the inital ROI
        frame_ROI = frame[initial_ROI[1]:initial_ROI[1]+initial_ROI[3], initial_ROI[0]:initial_ROI[0]+initial_ROI[2]]
        if not start_tracker:
            cv2.imshow('frame_ROI', frame_ROI)
            [boxes, confidences] = dnn_handler.find_person(frame_ROI)

            if len(boxes) != 0:
                start_tracker = True
                tracker.init(boxes[0])
                #TODO make draw function for dnn_handler
                cv2.imshow('frame_ROI', frame_ROI)

        if start_tracker:
            out_of_frame = tracker.update(frame)
            frame = tracker.draw(frame)

        #Scale frame to 50% for better overview
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('frame', frame)
        #Wait for 1 ms for keypress
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        if out_of_frame:
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

    # make spine fit to the x and y data





