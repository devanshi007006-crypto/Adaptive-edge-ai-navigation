#!/bin/bash

#######################################
# CONFIGURATION
#######################################
MEDIAMTX_COMMAND="/home/jetson/bme/mediamtx/mediamtx"    # Path to MediaMTX binary
PORT=8000                                               # Port used by MediaMTX
SUDO_PASSWORD="nvidia"                                  # sudo password
OBJECT_DETECTION_SCRIPT="/home/jetson/bme/object/object.py"

#######################################
# HELPER FUNCTIONS
#######################################

# Check if the port is in use
is_port_in_use() {
    # We redirect stderr to /dev/null to avoid printing errors
    sudo -S <<< "$SUDO_PASSWORD" lsof -i :"$PORT" &> /dev/null
}

# Free the port if in use
free_port() {
    if is_port_in_use; then
        echo "Port $PORT is in use. Freeing it..."
        PID=$(sudo -S <<< "$SUDO_PASSWORD" lsof -t -i :"$PORT")
        # Force kill
        sudo -S <<< "$SUDO_PASSWORD" kill -9 "$PID"
        echo "Port $PORT has been freed."
    fi
}

#######################################
# MEDIAMTX-RELATED
#######################################

# Start a single instance of MediaMTX (no loop in this function)
start_mediamtx() {
    echo "Starting MediaMTX..."
    cd /home/jetson/bme/mediamtx
    "$MEDIAMTX_COMMAND"
}

# Continuously run MediaMTX, restarting if it exits
start_mediamtx_loop() {
    while true; do
        # Ensure the port is free before starting
        free_port

        # Start MediaMTX (this will block until MediaMTX exits)
        start_mediamtx

        echo "MediaMTX has stopped. Restarting in 5 seconds..."
        sleep 5
    done
}

#######################################
# OBJECT-DETECTION-RELATED
#######################################

# Continuously run the object detection script, retrying if it fails
start_object_detection_loop() {
    while true; do
        echo "Starting object.py..."
        # Echo the sudo password and run python
        echo "$SUDO_PASSWORD" | sudo -S python3 "$OBJECT_DETECTION_SCRIPT"

        if [ $? -ne 0 ]; then
            echo "object.py failed. Retrying in 3 seconds..."
            sleep 3
        else
            echo "object.py exited successfully. Exiting loop..."
            # If your script can exit “successfully,” you might want to break,
            # or you can remove this logic to always keep it running.
            break
        fi

        echo "Restarting object.py in 5 seconds..."
        sleep 5
    done
}

#######################################
# CLEANUP / TRAPS
#######################################

cleanup() {
    echo "Script is exiting. Freeing up port $PORT (if needed)..."
    free_port
    # Optionally kill all background processes if you want a complete cleanup:
    echo "Killing all child processes..."
    kill 0
    echo "Cleanup complete."
}

# Trap signals so that cleanup runs on Ctrl+C or termination
trap cleanup SIGINT SIGTERM

#######################################
# MAIN EXECUTION
#######################################

# Start MediaMTX in the background
start_mediamtx_loop &
MEDIAMTX_PID=$!

# Start the object detection in the foreground (so when it dies, we notice)
start_object_detection_loop

# If we ever exit from object_detection_loop above, we proceed here:
echo "Object detection loop finished. Stopping MediaMTX..."
kill "$MEDIAMTX_PID"

echo "All done."
exit 0
