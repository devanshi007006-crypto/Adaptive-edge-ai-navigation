import sys, os
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2  # OpenCV for color format conversion
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy
import subprocess  # Needed for piping frames into ffmpeg
import logging 
import paho.mqtt.client as mqtt


# Parse the command line arguments
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="lines", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.4, help="minimum detection threshold to use") 

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# MQTT Broker Configuration
BROKER_ADDRESS = "localhost"
TOPIC = "object/detection"

# Initialize MQTT client
mqtt_client = mqtt.Client()
try:
    mqtt_client.connect(BROKER_ADDRESS, 1883)
    mqtt_client.loop_start()
    print(f"Connected to MQTT broker at {BROKER_ADDRESS}")
except Exception as e:
    print(f"Error connecting to MQTT broker: {e}")

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the color and depth streams (RGB and depth)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)    #1280, 720
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)    #640,480

# Start the pipeline
pipeline.start(config)

# Load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# Initialize the video output for rendering the image with detections
#output = videoOutput("webrtc://192.168.192.100:8554/output")
output = videoOutput("display://0")  # This will display the output on screen

# Start FFmpeg RTSP streaming
def start_ffmpeg_stream():
    ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "1280x720",
    "-r", "30",
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-profile:v", "baseline",  # Use baseline profile
    "-pix_fmt", "yuv420p",     # Use 4:2:0 chroma subsampling
    "-f", "rtsp",
    "-rtsp_transport", "tcp",  # Use TCP transport for RTSP
    "rtsp://127.0.0.1:8554/live"
]
    try:
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
        logging.info("FFmpeg RTSP stream started.")
        return process
    except Exception as e:
        logging.error(f"Failed to start FFmpeg RTSP stream: {e}")
        sys.exit(1)

# Start the FFmpeg pipeline
ffmpeg_process = start_ffmpeg_stream()

def stream_rtsp(frame):
    try:
        ffmpeg_process.stdin.write(frame.tostring())
    except Exception as e:
        logging.error(f"Failed to stream frame to FFmpeg: {e}")
        sys.exit(1)

# Capture frames until EOS or the user exits
try:
    while True:
        # Capture frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Failed to capture color or depth frame")
            continue  # Continue capturing frames if this happens
        
        # Convert the captured color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        #print(f"Captured frame: {color_image.shape}")  # Print shape of captured frame
        
        stream_rtsp(color_image)
        continue

        # Convert the BGR image to RGB format using OpenCV (this is crucial)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("RGB Image", color_image)
        # key=cv2.waitKey(1)

        # Convert the RGB image (numpy array) to CUDA memory that Jetson Inference can use
        cuda_image = cudaFromNumpy(rgb_image)
        # output.Render(cuda_image) 

        # Detect objects in the image (with overlay)
        detections = net.Detect(cuda_image, overlay=args.overlay)

        detected_objects = []
        detected_objects.append(f"detected {len(detections)} objects.")
        print(f"Detected {len(detections)} objects")


        # Loop over each detection and print the details
        for detection in detections:
            # Access the detection properties (Center, Width, Height)
            center_x, center_y = detection.Center  # Center is a tuple (x, y)
            width, height = detection.Width, detection.Height
            
            # Calculate the top-left corner (x1, y1) and bottom-right corner (x2, y2)
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            # Calculate the center of the bounding box (if needed)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Get the depth value at the center of the bounding box
            distance = depth_frame.get_distance(center_x, center_y)  # Distance in meters

            # Get confidence value and object label
            confidence = detection.Confidence
            label = detection.ClassID
            label_name = net.GetClassDesc(label)  # Get the class name based on class ID
            detected_objects.append(f"{label_name} at {distance:.2f} meters")

            # Prepare the overlay text with object label, distance, and confidence
            overlay_text = f"{confidence*100:.1f}% {label_name} at {distance:.2f}m"

            # Set the color for the bounding box
            box_color = (255, 255, 255)  # White color for the bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 1)
            cv2.putText(color_image, overlay_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1, cv2.LINE_AA)

            # Print the distance, label, and confidence
            print(f"Object: {label_name}, Distance: {distance:.2f} meters, Confidence: {confidence*100:.1f}%")

        # # Stream the frame to FFmpeg
        # stream_rtsp(color_image) # rgb_image , color_image , final_image , cuda_image 

        # Publish the detected objects to the MQTT broker
        if len(detected_objects) > 1:
            payload = "; ".join(detected_objects)
            print(f"Publishing: {payload}")
            mqtt_client.publish(TOPIC, payload)


        # # Convert the color image back to CUDA memory and render
        # final_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # cuda_image = cudaFromNumpy(final_image)
    
        # output.Render(cuda_image)  # Render the image with detections
        # output.SetStatus("Object Detection | Network FPS: {:.2f}".format(net.GetNetworkFPS()))  # Show FPS

        # # Print out performance info
        # net.PrintProfilerTimes()
    

        # Exit on input/output EOS
        if not output.IsStreaming():
            break

except KeyboardInterrupt:
    print("Exiting on user interrupt...")
    
finally:
    # Stop the RealSense pipeline after processing
    pipeline.stop()

    # Stop the FFmpeg process
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    print("FFmpeg process terminated.")

    # Disconnect from the MQTT broker
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Disconnected from MQTT broker.")

    # Clean up the CUDA memory and close the video output
    output.Close()
    del net
    print("Object detection completed.")

    # Exit the program
    sys.exit(0)
