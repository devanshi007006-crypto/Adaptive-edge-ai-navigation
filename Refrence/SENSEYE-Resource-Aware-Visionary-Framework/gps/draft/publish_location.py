import paho.mqtt.client as mqtt
import time

BROKER_ADDRESS = "localhost"  # Replace with Jetson IP if external
TOPIC = "location/live"

mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER_ADDRESS, 1883)

# List of sample locations
locations = [
    (37.7749, -122.4194),  # San Francisco
    (34.0522, -118.2437),  # Los Angeles
    (40.7128, -74.0060),   # New York
]

try:
    index = 0
    while True:  # Infinite loop for continuous updates
        lat, lon = locations[index]
        payload = f"{lat},{lon}"
        mqtt_client.publish(TOPIC, payload)
        print(f"Sent: {payload}")
        time.sleep(10)  # Publish every 1 second

        # Cycle through locations
        index = (index + 1) % len(locations)

except KeyboardInterrupt:
    print("Stopped publishing")
finally:
    mqtt_client.disconnect()
