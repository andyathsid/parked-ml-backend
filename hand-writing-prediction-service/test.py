import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://github.com/andyathsid/parked-ml-backend/blob/dev/hand-writing-prediction-service/sp1-P1.jpg?raw=true'}

result = requests.post(url, json=data).json()
print(result)