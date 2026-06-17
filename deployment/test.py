import requests

# ham
text = {
    "text": "sending test message"
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=text)
print(response.json())

#spam
text = {
    "text": "Congratulations! You have won a free iPhone. Click here now!"
}

response = requests.post(url, json=text)
print(response.json())

#ham
text = {
    "text": "U dun say so early hor... U c already then say..."
}

response = requests.post(url, json=text)
print(response.json())

#ham
text = {
    "text": "Hi. Wk been ok - on hols now! Yes on for a bit of a run. Forgot that i have hairdressers appointment at four so need to get home n shower beforehand. Does that cause prob for u?"
}

response = requests.post(url, json=text)
print(response.json())

#spam
text = {
    "text": "Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed å£1000 cash or å£5000 prize!"
}

response = requests.post(url, json=text)
print(response.json())