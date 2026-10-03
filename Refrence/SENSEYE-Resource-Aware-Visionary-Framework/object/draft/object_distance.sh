#!/bin/bash

# Run the Python script in sudo mode
echo "Starting object_distance.py..."
echo "nvidia" | sudo -S python3 /home/jetson/bme/object/object_distance.py
