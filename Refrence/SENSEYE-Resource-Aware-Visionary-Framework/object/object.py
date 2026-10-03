import sys, os
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy
import subprocess
import logging
import paho.mqtt.client as mqtt

# Parse the command line arguments
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-inception-v2", help="pre-trained model to load") # ssd-mobilenet-v2 
parser.add_argument("--overlay", type=str, default="lines", help="detection overlay flags")
parser.add_argument("--threshold", type=float, default=0.4, help="minimum detection threshold to use")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# MQTT Broker Configuration
BROKER_ADDRESS = "localhost"
TOPIC_QUERY_OBJ = "query/object"  # Topic to trigger object detection
TOPIC_QUERY_VLM = "query/vlm"  # Topic to trigger VLM
TOPIC_RESPONSE = "response/object"  # Topic to publish detection results
mqtt_client = None
detect_objects_flag = False  # Flag to trigger object detection when received MQTT message
detect_frame_flag = False  # Flag to trigger VLM when received MQTT message

# MQTT Callback function
def on_message(client, userdata, message):
    global detect_objects_flag
    global detect_frame_flag
    payload = message.payload.decode("utf-8").strip()
    if payload == "object":
        print("Object detection triggered via MQTT")
        detect_objects_flag = True

    if payload == "vlm":
        print("VLM triggered via MQTT")
        detect_frame_flag = True

# Initialize MQTT client
def init_mqtt_client():
    global mqtt_client
    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message
    try:
        mqtt_client.connect(BROKER_ADDRESS, 1883)
        mqtt_client.subscribe(TOPIC_QUERY_OBJ)  # Subscribe to the query topic
        mqtt_client.subscribe(TOPIC_QUERY_VLM)  # Subscribe to the query topic
        mqtt_client.loop_start()
        print(f"Connected to MQTT broker at {BROKER_ADDRESS}")
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the color and depth streams (RGB and depth)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# Initialize the video output for rendering the image with detections
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

# Capture frames and stream them continuously
try:
    init_mqtt_client()  # Initialize MQTT client

    while True:
        # Capture frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Failed to capture color or depth frame")
            continue

        # Convert the captured color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Stream the frame to FFmpeg continuously
        stream_rtsp(color_image)

        if detect_frame_flag:
            # save color image to  /home/jetson/bme/vlm  as frame.jpg
            cv2.imwrite("/home/jetson/bme/vlm/frame.jpg", color_image)
            print("Frame saved for VLM")
            detect_frame_flag = False

        if detect_objects_flag:
            print("Performing object detection...")

            # Convert the BGR image to RGB format using OpenCV (this is crucial)
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Convert the RGB image (numpy array) to CUDA memory that Jetson Inference can use
            cuda_image = cudaFromNumpy(rgb_image)

            # Detect objects in the image (with overlay)
            detections = net.Detect(cuda_image, overlay=args.overlay)

            detected_objects = []
            detected_objects.append(f"detected {len(detections)} objects.")
            print(f"Detected {len(detections)} objects")

            # Loop over each detection and print the details
            for detection in detections:
                center_x, center_y = detection.Center
                width, height = detection.Width, detection.Height
                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = int(center_x + width / 2)
                y2 = int(center_y + height / 2)

                # Calculate the center of the bounding box (if needed)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get the depth value at the center of the bounding box
                distance = depth_frame.get_distance(center_x, center_y)

                # Get confidence value and object label
                confidence = detection.Confidence
                label = detection.ClassID
                label_name = net.GetClassDesc(label)
                detected_objects.append(f"{label_name} at {distance:.2f} meters")

                # Prepare the overlay text with object label, distance, and confidence
                overlay_text = f"{confidence*100:.1f}% {label_name} at {distance:.2f}m"

                # Set the color for the bounding box
                box_color = (255, 255, 255)  # White color for the bounding box
                cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 1)
                cv2.putText(color_image, overlay_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1, cv2.LINE_AA)

                print(f"Object: {label_name}, Distance: {distance:.2f} meters, Confidence: {confidence*100:.1f}%")

            # Publish the detected objects to the MQTT broker on 'response/object'
            if len(detected_objects) > 1:
                payload = "; ".join(detected_objects)
                print(f"Publishing to {TOPIC_RESPONSE}: {payload}")
                mqtt_client.publish(TOPIC_RESPONSE, payload)

            detect_objects_flag = False  # Reset flag after detection

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
