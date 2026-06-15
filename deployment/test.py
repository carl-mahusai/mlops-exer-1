import requests

text = {
    "text": "sending test message"
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=text)
print(response.json())
