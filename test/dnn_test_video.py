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

    modules_abs_path = os.path.abspath("code/modules")
    dnn_models_abs_path = os.path.abspath("code/dnn_models")

    sys.path.append(modules_abs_path)
    sys.path.append(dnn_models_abs_path)

    from DnnHandler import DnnHandler



    logger = logging.getLogger(__name__)

    input_video = os.path.abspath("test/test_video/TestMovie3.avi")

    dnn_handler = DnnHandler()
    dnn_handler.init()

    #Read frames from video
    #Check if file exists
    if not os.path.isfile(input_video):
        print(f'Could not find file {input_video}')
        sys.exit(1)
    cap = cv2.VideoCapture(input_video)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video')
            break
        frame_count += 1
        modified_frame = frame
        if frame_count % 10 == 0:
            print(f'Looking for person in frame {frame_count}')
            modified_frame = dnn_handler.find_person(frame)
            modified_frame = cv2.resize(modified_frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('frame2', modified_frame)

        #Scale frame to 50% for better overview
        modified_frame = cv2.resize(modified_frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('frame', modified_frame)
        #Wait for 1 ms for keypress
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break