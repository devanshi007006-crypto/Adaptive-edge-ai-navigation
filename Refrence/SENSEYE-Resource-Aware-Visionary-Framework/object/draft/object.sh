#!/bin/bash

# Define the path to the object detection script
OBJECT_DETECTION_SCRIPT="/home/jetson/bme/object/draft/object_distance_rtsp.py"

# Function to start the object detection script
start_object_detection() {
    echo "Starting object_distance.py..."
    while true; do
        echo "nvidia" | sudo -S python3 $OBJECT_DETECTION_SCRIPT
        if [ $? -ne 0 ]; then
            echo "object_distance.py failed. Retrying in 2 seconds..."
            sleep 3  # Wait before retrying
        else
            echo "object_distance.py exited successfully. Exiting..."
            break
        fi
    done
}

# Main execution

# Start the object detection script
while true; do
    start_object_detection
    sleep 5
    echo "object_distance.py exited. Restarting in 5 seconds..."
    
done
