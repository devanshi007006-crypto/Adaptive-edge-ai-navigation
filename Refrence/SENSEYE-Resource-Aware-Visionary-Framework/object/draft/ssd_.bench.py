import time
from jetson_inference import detectNet
from jetson_utils import loadImage

# Load the object detection model
net = detectNet("ssd-inception-v2", threshold=0.5)

# Load the image
image_path = "dog.jpg"
image = loadImage(image_path)

# Run inference 10 times and measure time
num_runs = 10
times = []

for _ in range(num_runs):
    start_time = time.time()
    detections = net.Detect(image)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    times.append(inference_time)

# Compute average inference time
avg_inference_time = sum(times) / num_runs
print(f"Average Inference Time: {avg_inference_time:.2f} ms")
