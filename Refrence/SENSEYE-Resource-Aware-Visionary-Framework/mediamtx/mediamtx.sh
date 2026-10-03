#!/bin/bash

# Define the path to the MediaMTX executable and the port
MEDIAMTX_COMMAND="/home/jetson/bme/mediamtx/mediamtx"
PORT=8000
SUDO_PASSWORD="nvidia"

# Function to check if the port is in use
is_port_in_use() {
    sudo -S <<< "$SUDO_PASSWORD" lsof -i :$PORT > /dev/null 2>&1
}

# Function to free the port if in use
free_port() {
    if is_port_in_use; then
        echo "Port $PORT is in use. Freeing it..."
        PID=$(sudo -S <<< "$SUDO_PASSWORD" lsof -t -i :$PORT)
        sudo -S <<< "$SUDO_PASSWORD" kill -9 $PID
        echo "Port $PORT has been freed."
    fi
}

# Function to start MediaMTX
start_mediamtx() {
    echo "Starting MediaMTX..."
    $MEDIAMTX_COMMAND
}

# Trap to free the port on script exit
cleanup() {
    echo "Script is exiting. Freeing up port $PORT if necessary..."
    free_port
    echo "Cleanup complete. Exiting."
}

# Trap SIGINT and SIGTERM signals to run the cleanup function
trap cleanup SIGINT SIGTERM

# Main loop to restart MediaMTX on exit
while true; do
    free_port  # Ensure the port is free before starting
    start_mediamtx

    # Log that MediaMTX has exited and retry after 5 seconds
    echo "MediaMTX has stopped. Restarting in 5 seconds..."
    sleep 5
done
