import urllib.request
import json
import os

token = os.environ.get('HF_TOKEN')
headers = {'Authorization': f'Bearer {token}'} if token else {}
req = urllib.request.Request('https://huggingface.co/api/datasets/Yassaman/HEADS-UP/tree/main', headers=headers)
try:
    with urllib.request.urlopen(req) as resp:
        items = json.load(resp)
        for item in items:
            print(f"{item.get('type'):10s} {item.get('size', 0)/1e9:6.2f} GB  {item.get('path')}")
except Exception as e:
    print('Error:', e)
