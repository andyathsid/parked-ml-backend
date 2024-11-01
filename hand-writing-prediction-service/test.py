import requests

url = 'https://6suy7b2n3j.execute-api.ap-southeast-1.amazonaws.com/test/detect'

data = {'hw-url': 'https://github.com/andyathsid/parked-ml-dev/blob/main/hand-writing-detection/data/raw/NewHandPD/HealthySpiral/sp1-H1.jpg?raw=true'}

result = requests.post(url, json=data).json()
print(result)