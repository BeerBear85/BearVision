import cv2
import numpy as np
import os

class DnnHandler:
    """
    Wrapper around OpenCV's DNN module for running person-detection models.

    The class currently supports a subset of YOLOv8 ONNX models and exposes a
    minimal API used throughout the project.
    """

    def __init__(self, model_name):
        """Create a new handler for the given model.

        Args:
            model_name (str): Identifier of the ONNX model to load. Only a few
                predefined names (``yolov8n`` ... ``yolov8x``) are supported.

        This method does not load the network immediately to keep start-up
        times short; ``init`` must be called explicitly later on.
        """
        self.net = None
        # Determine the path relative to this file to avoid relying on the
        # current working directory, which makes the code more portable.
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_file_path, '../dnn_models')

        # Simple switch-case style dispatching to select the correct model
        # file. A dictionary would be more compact, but the explicit branches
        # make it easy to spot missing models.
        if model_name == 'yolov8n':
            self.model = os.path.join(model_path, 'yolov8n.onnx')
        elif model_name == 'yolov8s':
            self.model = os.path.join(model_path, 'yolov8s.onnx')
        elif model_name == 'yolov8m':
            self.model = os.path.join(model_path, 'yolov8m.onnx')
        elif model_name == 'yolov8l':
            self.model = os.path.join(model_path, 'yolov8l.onnx')
        elif model_name == 'yolov8x':
            self.model = os.path.join(model_path, 'yolov8x.onnx')
        else:
            # Failing fast here prevents later, harder-to-debug errors when the
            # model cannot be loaded.
            print('Error: Invalid model name')
            exit(1)

        self.confidence_threshold = 0.6

    def init(self):
        """Load the configured model into an OpenCV network.

        Returns:
            None
        """
        self.net = cv2.dnn.readNetFromONNX(self.model)

    def find_person(self, original_image):
        """Run inference and return detections for persons in the image.

        Args:
            original_image (numpy.ndarray): Image in BGR format.

        Returns:
            list: A two-element list ``[boxes, confidences]`` where ``boxes`` is
            a list of bounding boxes ``[x, y, w, h]`` in pixels and
            ``confidences`` the corresponding confidence values.
        """
        # OpenCV's DNN module expects a square image for YOLO models. Padding
        # avoids resizing, which would distort the aspect ratio and hurt
        # detection accuracy.
        (height, width) = original_image.shape[:2]
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Scale is needed to map detections from the 640x640 model input back
        # to the original resolution.
        scale = length / 640

        # Use blobFromImage for preprocessing because it performs normalization
        # and channel swapping in optimized C++ code.
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        self.net.setInput(blob)

        outputs = self.net.forward()

        # Model output is transposed for easier indexing: each row corresponds
        # to one detection candidate.
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores,
        # and class IDs.
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.confidence_threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Non-maximum suppression reduces overlapping detections, which keeps
        # downstream processing simple and avoids multiple boxes for the same
        # object.
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections_list = []
        for i in range(len(result_boxes)):
            # Handle different OpenCV versions: ensure index is an integer
            # In OpenCV 4.9+, result_boxes[i] may be a nested array/tuple
            index = result_boxes[i]
            if isinstance(index, (list, np.ndarray, tuple)):
                index = int(index[0])
            else:
                index = int(index)

            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'confidence': scores[index],
                'box': box,
                'scale': scale,
            }
            detections_list.append(detection)

        boxes_filtered = []
        confidences_filtered = []

        if len(detections_list) > 0:  # Ensure at least one detection exists
            for detection in detections_list:
                # Only proceed if the class label is 'person' (class ID 0 in
                # the COCO dataset).
                if detection['class_id'] == 0:
                    scaled_box = [round(box_coord * detection['scale']) for box_coord in detection['box']]
                    boxes_filtered.append(scaled_box)
                    confidences_filtered.append(detection['confidence'])

        return [boxes_filtered, confidences_filtered]
