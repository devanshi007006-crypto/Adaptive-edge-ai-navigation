import sys
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage())
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load")
parser.add_argument("--threshold", type=float, default=0.4, help="minimum detection threshold to use")
args = parser.parse_args()

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

try:
    pipeline.start(config)
    logging.info("RealSense pipeline started successfully.")
except Exception as e:
    logging.error(f"Failed to start RealSense pipeline: {e}")
    sys.exit(1)

# Initialize detectNet
try:
    net = detectNet(args.network, sys.argv, args.threshold)
    logging.info("detectNet initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize detectNet: {e}")
    sys.exit(1)

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

try:
    while True:
        # Capture frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            logging.warning("No color frame received.")
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        logging.debug(f"Captured frame shape: {color_image.shape}")

        # Perform object detection
        try:
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cuda_image = cudaFromNumpy(rgb_image)
            detections = net.Detect(cuda_image)
            logging.debug(f"Detections: {detections}")
        except Exception as e:
            logging.error(f"Error during object detection: {e}")
            continue

        # Draw detections on the image
        for detection in detections:
            center_x, center_y = detection.Center
            width, height = detection.Width, detection.Height
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            label_name = net.GetClassDesc(detection.ClassID)
            confidence = detection.Confidence * 100
            overlay_text = f"{label_name} ({confidence:.1f}%)"

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(color_image, overlay_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Write frame to FFmpeg stdin
        try:
            ffmpeg_process.stdin.write(color_image.tobytes())
        except Exception as e:
            logging.error(f"Failed to write frame to FFmpeg pipeline: {e}")
            break

except KeyboardInterrupt:
    logging.info("Exiting on user interrupt.")
finally:
    # Stop pipeline and FFmpeg process
    logging.info("Stopping pipelines.")
    pipeline.stop()
    if ffmpeg_process:
        ffmpeg_process.terminate()
