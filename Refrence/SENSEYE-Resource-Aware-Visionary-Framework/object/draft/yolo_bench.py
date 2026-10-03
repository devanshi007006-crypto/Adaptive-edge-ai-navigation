import time
from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("yolo11m.pt")

# Export the model to TensorRT
#model.export(format="engine")  # FP32
model.export(format="engine", half=True)  # FP16
#model.export(format="engine", int8=True)  # # INT8

# Load the exported TensorRT model
trt_model = YOLO("yolo11m.engine")

# Run inference with timing
image_path = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"

n=1
while n <=10 :
    results = trt_model(image_path)  # Run inference
    n+=1
    # Process results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class index
            confidence = float(box.conf[0])  # Confidence score
            label = trt_model.names[class_id]  # Get class label
            print(f"Detected: {label}, Confidence: {confidence:.4f}")

