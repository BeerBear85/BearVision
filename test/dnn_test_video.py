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
    import time

    start_time = time.time()
    modules_abs_path = os.path.abspath(os.path.join("code", "modules"))
    dnn_models_abs_path = os.path.abspath(os.path.join("code", "dnn_models"))

    sys.path.append(modules_abs_path)
    sys.path.append(dnn_models_abs_path)

    from DnnHandler import DnnHandler


    logger = logging.getLogger(__name__)

    input_video = os.path.abspath(os.path.join("test", "input_video", "TestMovie1.mp4"))

    dnn_handler = DnnHandler("yolov8s")
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
            [boxes, confidences] = dnn_handler.find_person(frame)

            for i, box in enumerate(boxes):
                # Draw bounding box for the object
                (x, y) = (box[0], box[1])
                (w, h) = (box[2], box[3])
                cv2.rectangle(modified_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                modified_frame = cv2.resize(modified_frame, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('frame2', modified_frame)
                # Draw label text with confidence score
                label = "Person: {:.2f}%".format(confidences[i] * 100)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y_label = max(y, labelSize[1])
                cv2.rectangle(modified_frame, (x, y_label - labelSize[1] - 10), (x + labelSize[0], y_label + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(modified_frame, label, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                print(f'Found person in frame {frame_count} with confidence {confidences[i] * 100:.2f}%')
                

        #Scale frame to 50% for better overview
        modified_frame = cv2.resize(modified_frame, (0, 0), fx=0.5, fy=0.5)
        #cv2.imshow('frame', modified_frame)
        #Wait for 1 ms for keypress
        #if cv2.waitKey(50) & 0xFF == ord('q'):
        #    break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")