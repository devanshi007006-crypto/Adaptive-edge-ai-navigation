#!/bin/bash

# Function to run location.py in the background and monitor it
start_location() {
    while true; do
        echo "Starting location.py..."
        python3 /home/jetson/bme/gps/location.py
        echo "location.py crashed. Restarting in 5 seconds..."
        sleep 5
    done
}

# Function to run mqtt_server.py in the foreground and monitor it
start_mqtt_server() {
    while true; do
        echo "Starting mqtt_server.py..."
        python3 /home/jetson/bme/gps/mqtt_server.py
        echo "mqtt_server.py crashed. Restarting in 5 seconds..."
        sleep 5
    done
}

# Start location.py in the background
start_location &
LOCATION_PID=$!

# Start mqtt_server.py in the foreground
start_mqtt_server

# Cleanup: Kill location.py when mqtt_server.py exits
echo "Stopping location.py..."
kill $LOCATION_PID
echo "Done."