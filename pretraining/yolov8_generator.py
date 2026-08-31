from ultralytics import YOLO
import os

input_model_list = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
base_dir = os.path.dirname(os.path.abspath(__file__))

# input_model_list = [input_model_list[0]] #only take one model

for input_model in input_model_list:
    print("Beginning converting model: ", input_model)

    input_model_path = os.path.join(base_dir, input_model)
    output_model_path = input_model_path.replace(".pt", ".onnx")
    model = YOLO(input_model_path)

    #Delete old output model
    if os.path.exists(output_model_path):
        os.remove(output_model_path)
    path = model.export(format="onnx", imgsz=640, opset=12)  # export the model to ONNX format
    os.rename(path, output_model_path)
    print("Finish converting model: ", output_model_path)
    