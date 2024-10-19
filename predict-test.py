
import requests


url = 'http://localhost:9696/predict'

file_path = './data/raw/FB1cdaopmoe67M2605161917.wav'
with open(file_path, 'rb') as file:
    files = {'file': file}  
    response = requests.post(url, files=files)

response_json = response.json()
print(response_json)
