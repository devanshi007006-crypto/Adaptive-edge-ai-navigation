#!/bin/bash
#gst-rtsp-server-1.0 -p 8554
# go to location
cd ~/jetson-inference/build/aarch64/bin

# Go to the specified location
cd ~/jetson-inference/build/aarch64/bin

# Loop to rerun the command if it fails
while true; do
    gst-launch-1.0 v4l2src device=/dev/video4 ! tee name=t \
        t. ! queue ! videoconvert ! x264enc ! rtph264pay ! rtspclientsink location=rtsp://192.168.192.100:8554/stream1 \
        t. ! queue ! videoconvert ! x264enc ! rtph264pay ! rtspclientsink location=rtsp://192.168.192.100:8555/stream2
    if [ $? -eq 0 ]; then
        break
    else
        echo "Command failed, retrying..."
        sleep 2  # Optional: wait for 2 seconds before retrying
    fi
done