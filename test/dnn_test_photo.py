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

    modules_abs_path = os.path.abspath("code/modules")
    dnn_models_abs_path = os.path.abspath("code/dnn_models")

    sys.path.append(modules_abs_path)
    sys.path.append(dnn_models_abs_path)

    from DnnHandler import DnnHandler
    logger = logging.getLogger(__name__)

    input_image_list = list()
    input_image_list.append(os.path.abspath("test/images/test_image_1.jpg"))
    input_image_list.append(os.path.abspath("test/images/test_image_2.jpg"))
    input_image_list.append(os.path.abspath("test/images/test_image_3.jpg"))
    input_image_list.append(os.path.abspath("test/images/test_image_4.jpg"))
    input_image_list.append(os.path.abspath("test/images/test_image_5.jpg"))

    #input_image_list = [input_image_list[2]]

    for image_path in input_image_list:





        #Check if file exists
        if not os.path.isfile(image_path):
            print(f'Could not find file {image_path}')
            sys.exit(1)
        frame = cv2.imread(image_path)

        #if image contains "test_image_2", scale to 50%
        if "test_image_2" in image_path:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


        dnn_handler = DnnHandler()
        dnn_handler.init()

        [boxes, confidences] = dnn_handler.find_person(frame)

        for i, box in enumerate(boxes):
            # Draw bounding box for the object
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label text with confidence score
            label = "Person: {:.2f}%".format(confidences[i] * 100)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(y, labelSize[1])
            cv2.rectangle(frame, (x, y_label - labelSize[1] - 10), (x + labelSize[0], y_label + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f'Found person in file {image_path} with confidence {confidences[i] * 100:.2f}%')

        cv2.imshow('frame', frame)
        cv2.waitKey(2000)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")