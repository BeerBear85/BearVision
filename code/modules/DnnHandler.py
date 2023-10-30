import cv2
import numpy as np
import os

class DnnHandler:
    def __init__(self):
        self.net = None
        #Abs path of the folder of the current file
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_file_path, '../dnn_models')

        #self.model = os.path.join(model_path, 'yolov8n.onnx')
        #self.model = os.path.join(model_path, 'yolov8s.onnx')
        self.model = os.path.join(model_path, 'yolov8m.onnx')
        #self.model = os.path.join(model_path, 'yolov8l.onnx')
        #self.model = os.path.join(model_path, 'yolov8x.onnx')
        self.threshold = 0.6

    def init(self):
        #self.net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_weights)
        self.net = cv2.dnn.readNetFromONNX(self.model)

    def find_person(self, original_image):
        # Load the image
        (height, width) = original_image.shape[:2]

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        self.net.setInput(blob)

        # Perform inference
        outputs = self.net.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections_list = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                #'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections_list.append(detection)

        #Create array of boxes and confidences after filtering
        boxes_filtered = []
        confidences_filtered = []
        
        if len(detections_list) > 0: # Ensure at least one detection exists
            for detection in detections_list:
                # Only proceed if the class label is 'person' (class ID 0 in COCO dataset)
                if detection['class_id'] == 0:
                    scaled_box = [round(box_coord*detection['scale']) for box_coord in detection['box']]
                    boxes_filtered.append(scaled_box)
                    confidences_filtered.append(detection['confidence'])

        return [boxes_filtered, confidences_filtered]
