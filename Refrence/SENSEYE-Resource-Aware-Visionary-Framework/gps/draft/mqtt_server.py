import paho.mqtt.client as mqtt
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import datetime

# Flask and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Latest location data
latest_location = {
    "latitude": 0,
    "longitude": 0,
    "method": "Unknown",
    "location": "Unknown Location",
    "last_update": datetime.datetime.now(),
}

# MQTT Broker details
BROKER_ADDRESS = "localhost"
TOPIC = "location/live"

# Function to calculate "time ago"
def time_ago(last_update):
    now = datetime.datetime.now()
    diff = now - last_update
    seconds = diff.total_seconds()
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    else:
        return f"{int(seconds // 3600)} hours ago"

def on_message(client, userdata, message):
    global latest_location
    payload = message.payload.decode("utf-8")
    print(f"MQTT Message Received: {payload}")

    try:
        # Split payload into parts
        parts = payload.split(",", maxsplit=3)  # Limit to 4 parts: lat, lon, method, location_name
        if len(parts) < 3:
            raise ValueError("Payload does not contain enough values")

        # Parse latitude, longitude, method, and location name
        lat = parts[0]
        lon = parts[1]
        method = parts[2]
        location_name = parts[3] if len(parts) > 3 else "Unknown Location"

        # Update the latest location data
        latest_location["latitude"] = float(lat)
        latest_location["longitude"] = float(lon)
        latest_location["method"] = method
        latest_location["location"] = location_name
        latest_location["last_update"] = datetime.datetime.now()

        # Prepare data for WebSocket emission
        data_to_emit = {
            "latitude": latest_location["latitude"],
            "longitude": latest_location["longitude"],
            "method": latest_location["method"],
            "location": latest_location["location"],
            "last_update": time_ago(latest_location["last_update"]),
        }

        # Emit data via WebSocket
        socketio.emit("location_update", data_to_emit)
    except Exception as e:
        print(f"Error parsing MQTT message: {e}")

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message

try:
    mqtt_client.connect(BROKER_ADDRESS, 1883)
    print(f"Connected to MQTT broker at {BROKER_ADDRESS}")
    mqtt_client.subscribe(TOPIC)
    mqtt_client.loop_start()
except Exception as e:
    print(f"Error connecting to MQTT broker: {e}")

# Serve the frontend
@app.route("/")
def index():
    return render_template("index_live.html")

# Start Flask and SocketIO server
if __name__ == "__main__":

    print("Starting Flask and SocketIO server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
