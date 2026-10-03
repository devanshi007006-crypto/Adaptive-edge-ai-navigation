from scapy.all import *
import subprocess

def get_wifi_networks():
    networks = []
    try:
        # Use Linux `iwlist` to scan Wi-Fi networks
        result = subprocess.check_output(["sudo", "iwlist", "wlan0", "scan"], text=True)
        for line in result.split("\n"):
            if "ESSID" in line:
                ssid = line.split(":")[1].strip('"')
                networks.append(ssid)
    except Exception as e:
        print(f"Error scanning Wi-Fi networks: {e}")
    return networks

wifi_networks = get_wifi_networks()
print("Nearby Wi-Fi Networks:", wifi_networks)
