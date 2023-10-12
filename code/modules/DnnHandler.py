import cv2
import numpy as np
import os

class DnnHandler:
    def __init__(self):
        self.net = None
        #Abs path of the folder of the current file
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.model_cfg = os.path.join(current_file_path, '../dnn_models/yolov3_608.cfg')
        self.model_weights = os.path.join(current_file_path, '../dnn_models/yolov3_608.weights')
        self.threshold = 0.7

    def init(self):
        self.net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_weights)

    def find_person(self, image):
        # Load the image
        (h, w) = image.shape[:2]

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Get the output layer names
        layer_names = self.net.getLayerNames()

        # Get the indices of the output layers
        unconnected_out_layers_indices = self.net.getUnconnectedOutLayers()

        # Get the names of the output layers
        output_layers = []
        for layer_idx in unconnected_out_layers_indices: 
            layer_name = layer_names[layer_idx - 1]
            output_layers.append(layer_name)


        # Perform a forward pass
        layer_outputs = self.net.forward(output_layers)

        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Loop over each of the layer outputs
        for output in layer_outputs:
            # Loop over each of the detections
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.threshold:  # You can adjust this threshold
                    # Scale the bounding box coordinates back to the size of the image
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # You can adjust these thresholds

        #Create array of boxes and confidences after filtering
        boxes_filtered = []
        confidences_filtered = []
        
        if len(idxs) > 0: # Ensure at least one detection exists
            for i in idxs.flatten():
                # Only proceed if the class label is 'person' (class ID 0 in COCO dataset)
                if class_ids[i] == 0:
                    boxes_filtered.append(boxes[i])
                    confidences_filtered.append(confidences[i])

        return [boxes_filtered, confidences_filtered]

