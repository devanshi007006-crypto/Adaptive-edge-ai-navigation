import Jetson.GPIO as GPIO
import serial
import time
import paho.mqtt.client as mqtt
import requests
from datetime import datetime

# MQTT Broker Configuration
BROKER_ADDRESS = "localhost"  # Replace with the actual broker address
TOPIC = "location/live"

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER_ADDRESS, 1883)

# Serial Port Configuration
ser = serial.Serial('/dev/ttyTHS1', 115200)
ser.flushInput()

# GPIO Configuration
powerKey = 6

# Function to send AT commands
def sendAt(command, back, timeout):
    ser.write((command + '\r\n').encode())
    time.sleep(timeout)
    if ser.inWaiting():
        rec_buff = ser.read(ser.inWaiting())
        if back in rec_buff.decode():
            return rec_buff.decode()
    return None

# Function to convert DDMM.MMMM to decimal degrees
def convert_to_decimal(degrees_minutes, is_longitude=False):
    try:
        degrees_length = 3 if is_longitude else 2
        degrees = int(degrees_minutes[:degrees_length])
        minutes = float(degrees_minutes[degrees_length:])
        return degrees + (minutes / 60)
    except Exception as e:
        print(f"Error converting to decimal: {e}")
        return 0.0

# Function to get GPS location
def getGpsPosition():
    print("Attempting to get GPS location...")
    sendAt("AT+CGPS=1,1", "OK", 1)
    time.sleep(3)
    gps_info = sendAt("AT+CGPSINFO", "+CGPSINFO: ", 1)
    if gps_info:
        try:
            data = gps_info.split(":")[1].strip()
            print(f"Raw data: {data}")
            # Check if GPS data is valid
            if data and data != ",,,,,,,,": 
                lat_raw, lon_raw = data.split(",")[0], data.split(",")[2]
                # Ensure lat_raw and lon_raw are not empty
                if lat_raw and lon_raw:
                    lat = convert_to_decimal(lat_raw)  # Latitude conversion
                    lon = convert_to_decimal(lon_raw, is_longitude=True)  # Longitude conversion
                    place_name = getPlaceName(lat, lon)
                    print(f"GPS Location: {lat:.7f}, {lon:.7f}")
                    return lat, lon, place_name, "GPS"
        except Exception as e:
            print(f"Error parsing GPS data: {e}")
    print("No valid GPS data available.")
    return None


# Function to get location from IP
def getIpLocation():
    print("Falling back to IP-based location...")
    try:
        ip_response = requests.get("https://api.ipify.org?format=json", timeout=5)
        ip = ip_response.json().get("ip")
        loc_response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        if loc_response.status_code == 200:
            loc_data = loc_response.json()
            lat, lon = loc_data.get("lat"), loc_data.get("lon")
            place_name = loc_data.get("city", "Unknown")
            print(f"IP Location: {lat}, {lon}")
            return lat, lon, place_name, "IP"
    except Exception as e:
        print(f"Error getting IP location: {e}")
    return None

# Function to get place name using reverse geocoding
def getPlaceName(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        response = requests.get(url, headers={"User-Agent": "BMEProject/1.0"}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            full_address = data.get("display_name", "Unknown Location")
            return ", ".join(full_address.split(", ")[:4])
    except Exception as e:
        print(f"Error getting place name: {e}")
    return "Unknown Location"


# Power on the SIM7600X module
def powerOn():
    print("Powering on SIM7600X...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(powerKey, GPIO.OUT)
    GPIO.output(powerKey, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(powerKey, GPIO.LOW)
    time.sleep(10)
    print("SIM7600X is ready.")

# Power off the SIM7600X module
def powerDown():
    print("Powering off SIM7600X...")
    GPIO.output(powerKey, GPIO.HIGH)
    time.sleep(3)
    GPIO.output(powerKey, GPIO.LOW)
    time.sleep(10)
    print("SIM7600X is off.")

# Function to calculate time elapsed
def time_ago(timestamp):
    now = datetime.now()
    diff = now - timestamp
    seconds = diff.total_seconds()
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    return f"{int(seconds // 3600)} hours ago"


# Main publishing loop
def publish_location():
    try:
        powerOn()
        while True:
            # Try to get GPS location
            location = getGpsPosition()
            if not location:
                # Fallback to IP-based location if GPS is unavailable or invalid
                location = getIpLocation()

            if location:
                lat, lon, place_name, method = location
                last_update = datetime.now()
                payload = f"{lat},{lon},{method},{place_name}"
                mqtt_client.publish(TOPIC, payload)
                print(f"{place_name} | {method} | Last update: {time_ago(last_update)}")
                #save_location_to_file(place_name, method)
            else:
                print("Failed to obtain location.")

            # Delay before retrying
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping location publishing.")
    finally:
        powerDown()
        mqtt_client.disconnect()
        GPIO.cleanup()

# Start publishing
if __name__ == "__main__":
    publish_location()
