import sys
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2  # OpenCV for color format conversion
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy

# Parse the command line arguments
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="/dev/video4", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="lines,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream (RGB)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# Initialize the video output for rendering the image with detections
output = videoOutput("display://0")  # This will display the output on screen

# Capture frames until EOS or the user exits
while True:
    # Capture frames from the RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        print("Failed to capture color frame")
        continue  # Continue capturing frames if this happens
    
    # Convert the captured color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())
    print(f"Captured frame: {color_image.shape}")  # Print shape of captured frame

    # Convert the BGR image to RGB format using OpenCV
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Convert the RGB image (numpy array) to CUDA memory that Jetson Inference can use
    cuda_image = cudaFromNumpy(rgb_image)

    # Detect objects in the image (with overlay)
    detections = net.Detect(cuda_image, overlay=args.overlay)

    print(f"Detected {len(detections)} objects")

    # Loop over each detection and print the details
    for detection in detections:
        print(f"Detected object: {detection.ClassID}, Confidence: {detection.Confidence}")

    # Render the image with detection overlays
    output.Render(cuda_image)  # Render the image with detections
    output.SetStatus("Object Detection | Network FPS: {:.2f}".format(net.GetNetworkFPS()))  # Show FPS

    # Print out performance info
    net.PrintProfilerTimes()

    # Exit on input/output EOS
    if not output.IsStreaming():
        break

# Stop the RealSense pipeline after processing
pipeline.stop()
