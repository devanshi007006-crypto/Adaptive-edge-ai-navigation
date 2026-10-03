import requests

def get_location_google(api_key):
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
    response = requests.post(url, json={"considerIp": "true"})
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.json())
        return None

api_key = "AIzaSyBJXVbFmM5BfgkkKO3ZMAHcBnvQxjbFJ1A"
location = get_location_google(api_key)
if location:
    print("Latitude:", location["location"]["lat"])
    print("Longitude:", location["location"]["lng"])
