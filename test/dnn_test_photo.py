import cv2
import numpy as np

input_image = '../test_image_2.jpg'

# Load YOLOv3 model
model_cfg = 'yolov3_608.cfg'
model_weights = 'yolov3_608.weights'
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

# Load the image
image = cv2.imread(input_image)
(h, w) = image.shape[:2]

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get the output layer names
layer_names = net.getLayerNames()

# Get the indices of the output layers
unconnected_out_layers_indices = net.getUnconnectedOutLayers()

# Get the names of the output layers
output_layers = []
for layer_idx in unconnected_out_layers_indices: 
    layer_name = layer_names[layer_idx - 1]
    output_layers.append(layer_name)


# Perform a forward pass
layer_outputs = net.forward(output_layers)

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
        if confidence > 0.5:  # You can adjust this threshold
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

# Ensure at least one detection exists
if len(idxs) > 0:
    for i in idxs.flatten():
        # Only proceed if the class label is 'person' (class ID 0 in COCO dataset)
        if class_ids[i] == 0:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label text with confidence score
            label = "Person: {:.2f}%".format(confidences[i] * 100)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(y, labelSize[1])
            cv2.rectangle(image, (x, y_label - labelSize[1] - 10), (x + labelSize[0], y_label + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Show the output image

# Scale the image
scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

cv2.imshow("Image", scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
