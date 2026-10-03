import requests

def get_location_ipstack(api_key):
    url = f"http://api.ipstack.com/check?access_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.json())
        return None

api_key = "9d1956cd119ee495983b221cef73e2a0"
location = get_location_ipstack(api_key)
if location:
    print("City:", location["city"])
    print("Region:", location["region_name"])
    print("Country:", location["country_name"])
    print("Latitude:", location["latitude"])
    print("Longitude:", location["longitude"])
