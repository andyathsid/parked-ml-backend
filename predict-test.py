import requests
import json
import base64

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

file_path = './data/raw/FB1cdaopmoe67M2605161917.wav'

with open(file_path, 'rb') as file:
    file_content = file.read()

encoded_file = base64.b64encode(file_content).decode('utf-8')

payload = {
    'file': encoded_file,
    'filename': 'FB1cdaopmoe67M2605161917.wav'  
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print(response.json())
