import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://github.com/andyathsid/parked-ml-backend/blob/dev-alt/hand-writing-prediction-service/sp1-P1.jpg?raw=true'}

result = requests.post(url, json=data).json()
print(result)