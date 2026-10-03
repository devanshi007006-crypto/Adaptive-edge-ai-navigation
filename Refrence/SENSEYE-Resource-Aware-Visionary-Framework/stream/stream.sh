#!/bin/bash

##############################################################################
# CONFIGURATION
##############################################################################
cd /home/jetson/bme/stream || exit 1

# Variables
timeout_count=0
max_timeouts=15

# Flag for which source to use
use_rtsp=true   # start with RTSP

##############################################################################
# FUNCTIONS
##############################################################################
# 1) RTSP streaming
stream_rtsp() {
    video-viewer --headless \
        --ssl-key=key.pem \
        --ssl-cert=cert.pem \
        --input-codec=h264 \
        rtsp://127.0.0.1:8554/live \
        webrtc://@:8555/output
}

# 2) Webcam streaming
stream_webcam() {
    video-viewer --headless \
        --ssl-key=key.pem \
        --ssl-cert=cert.pem \
        /dev/video4 \
        webrtc://@:8555/output
}

# 3) Monitor the output of video-viewer for timeouts or failure
#    - Reads lines from stdin (the output of video-viewer).
#    - If max_timeouts is exceeded or input stream fails, return 1 => indicates a source switch.
monitor() {
    while IFS= read -r line; do
        echo "$line"

        # Condition A: Timeout messages
        if [[ "$line" == *"timeout occurred waiting for the next image buffer"* ]]; then
            ((timeout_count++))
            echo "Timeout #$timeout_count"

            if [[ $timeout_count -ge $max_timeouts ]]; then
                echo "Reached $max_timeouts timeouts. Need to switch source..."
                # Kill video-viewer so we can restart
                pkill -f "video-viewer"
                return 1
            fi

        # Condition B: "failed to create input stream" => e.g. /dev/video4 doesn't exist
        elif [[ "$line" == *"failed to create input stream"* ]]; then
            echo "Input stream creation failed. Switching source..."
            pkill -f "video-viewer"
            return 1

        else
            # Reset counter if not a timeout line
            timeout_count=0
        fi
    done

    # If video-viewer exits normally or is interrupted:
    return 0
}

##############################################################################
# MAIN LOOP
##############################################################################
while true; do
    if [ "$use_rtsp" = true ]; then
        echo "=== Starting RTSP stream ==="
        stream_rtsp 2>&1 | monitor
    else
        echo "=== Starting Webcam (/dev/video4) stream ==="
        stream_webcam 2>&1 | monitor
    fi

    exit_code=$?

    # If monitor() returned 1 => switch source
    if [ "$exit_code" -eq 1 ]; then
        if [ "$use_rtsp" = true ]; then
            use_rtsp=false
        else
            use_rtsp=true
        fi
        # Reset the timeout counter after switching
        timeout_count=0
    fi

    echo "Retrying in 5 seconds..."
    sleep 5
done