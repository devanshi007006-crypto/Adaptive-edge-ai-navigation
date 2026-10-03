import sys
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2  # OpenCV for color format conversion
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy
from gtts import gTTS
import subprocess
import threading
import os
from queue import Queue
import time

# Configure PulseAudio for root
os.environ["PULSE_SERVER"] = "/run/user/1000/pulse/native"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

# Create a TTS task queue
tts_queue = Queue()

def parse_arguments():
    """
    Parse the command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Locate objects in a live camera stream using an object detection DNN.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage(),
    )

    parser.add_argument("input", type=str, default="/dev/video4", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="webrtc://@:8555/object", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="Pre-trained model to load.")
    parser.add_argument("--overlay", type=str, default="box,labels,conf", help="Detection overlay flags.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Minimum detection threshold to use.")

    try:
        return parser.parse_known_args()[0]
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

def initialize_realsense():
    """
    Initialize the Intel RealSense pipeline for capturing frames.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the color and depth streams
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipeline.start(config)
    return pipeline

def text_to_speech(text, audio_file="object_detected.mp3"):
    """
    Convert text to speech and save it as an MP3 file.
    """
    try:
        tts = gTTS(text, lang="en")
        tts.save(audio_file)
        print(f"Audio file saved as {audio_file}")
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

def trigger_audio_playback(audio_file):
    """
    Trigger audio playback using play_audio.py as a subprocess.
    """
    try:
        subprocess.run(
            ["sudo", "-u", "jetson", "python3", "play_audio.py", audio_file],
            env=os.environ,
            check=True,
        )
        print("Audio played successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while trying to play audio: {e}")
    except Exception as e:
        print(f"Unexpected error in audio playback: {e}")

def speak_text_blocking(text):
    """
    Perform TTS and audio playback in a blocking manner.
    """
    audio_file = "object_detected.mp3"
    try:
        text_to_speech(text, audio_file)
        trigger_audio_playback(audio_file)
    except Exception as e:
        print(f"Error in TTS or audio playback: {e}")

def tts_worker():
    """
    Worker thread to process TTS tasks sequentially.
    """
    while True:
        text = tts_queue.get()
        if text is None:
            break
        speak_text_blocking(text)
        tts_queue.task_done()

def speak_text_nonblocking(text):
    """
    Add the TTS task to the queue to ensure serialized playback.
    """
    tts_queue.put(text)
    print(f"Added TTS task to the queue: {text}")

def process_detections(detections, depth_frame, net, color_image):
    """
    Process detections and handle TTS and overlay rendering.
    """
    for detection in detections:
        center_x, center_y = detection.Center
        width, height = detection.Width, detection.Height
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        # Get the depth value at the center of the bounding box
        distance = depth_frame.get_distance(int(center_x), int(center_y))
        label_name = net.GetClassDesc(detection.ClassID)
        confidence = detection.Confidence

        # Prepare the overlay text
        overlay_text = f"{confidence*100:.1f}% {label_name} at {distance:.2f}m"

        # Draw the bounding box and label
        box_color = (255, 255, 255)
        cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(color_image, overlay_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Speak the detection if close enough
        if 0 < distance <= 2:
            speech_text = f"{label_name} detected at {distance:.2f} meters."
            print(speech_text)
            speak_text_nonblocking(speech_text)

def main():
    args = parse_arguments()

    # Initialize the RealSense pipeline
    pipeline = initialize_realsense()

    # Load the object detection network
    net = detectNet(args.network, sys.argv, args.threshold)

    # Initialize the video output for rendering the image with detections
    output = videoOutput("display://0")  # This will display the output on screen

    # Start the TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    try:
        last_detection_time = 0
        detection_interval = 0.5  # Minimum time between detections (seconds)

        while True:
            # Capture frames from the RealSense camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Skipping empty frame")
                continue

            # Convert the captured color frame to a numpy array
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cuda_image = cudaFromNumpy(rgb_image)

            # Detect objects in the image
            detections = net.Detect(cuda_image, overlay=args.overlay)

            # Process detections and draw bounding boxes
            if time.time() - last_detection_time >= detection_interval:
                process_detections(detections, depth_frame, net, color_image)
                last_detection_time = time.time()

            # Render the image
            final_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cuda_image = cudaFromNumpy(final_image)
            output.Render(cuda_image)
            output.SetStatus("Object Detection | Network FPS: {:.2f}".format(net.GetNetworkFPS()))

            # Exit if the output is not streaming
            if not output.IsStreaming():
                break

    finally:
        # Stop the RealSense pipeline after processing
        pipeline.stop()

        # Stop the TTS worker thread
        tts_queue.put(None)
        tts_thread.join()

if __name__ == "__main__":
    main()
