import requests

def get_public_ip():
    response = requests.get("https://api.ipify.org?format=json")
    return response.json().get("ip")

ip_address = get_public_ip()
print("Public IP Address:", ip_address)

def get_location_by_ip(ip):
    url = f"http://ip-api.com/json/{ip}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.json())
        return None

ip = get_public_ip()
location = get_location_by_ip(ip)
if location:
    print("City:", location.get("city"))
    print("Region:", location.get("regionName"))
    print("Country:", location.get("country"))
    print("Latitude:", location.get("lat"))
    print("Longitude:", location.get("lon"))
