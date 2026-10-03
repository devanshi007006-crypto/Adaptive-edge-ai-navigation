#!/bin/bash

# Initialize a counter for timeouts
timeout_count=0
max_timeouts=10

# Infinite loop to retry the process if timeout occurs
while true; do
    # Start the video viewer command and monitor its output
    cd /home/jetson/bme/stream
    video-viewer --headless --ssl-key=key.pem --ssl-cert=cert.pem --input-codec=h264 rtsp://127.0.0.1:8554/live webrtc://@:8555/output 2>&1 | while IFS= read -r line
    do
        # Log the line to the console for debugging
        echo "$line"

        # Check if the line contains the timeout message
        if [[ "$line" == *"timeout occurred waiting for the next image buffer"* ]]; then
            # Increment the timeout counter
            ((timeout_count++))

            # If we reach the max timeouts, exit the script
            if [[ $timeout_count -ge $max_timeouts ]]; then
                echo "Timeout limit reached, exiting."
                # Kill the background process if it's running
                pkill -f "video-viewer"
                exit 1
            fi
        else
            # Reset the timeout counter if a different message appears
            timeout_count=0
        fi
    done

    # If we get here, it means the video-viewer has finished or was interrupted
    # Wait 5 seconds before retrying
    echo "Retrying in 5 seconds..."
    sleep 5
done
