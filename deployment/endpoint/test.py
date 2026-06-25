import requests

url = 'http://localhost:9696/predict'

# ham
text = {
    "text": "sending test message"
}
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

ham_message_array = [
    "Hey, are we still meeting for lunch at 12:30?",
    "Don't forget to bring your laptop for today's meeting.",
    "I'm running about 10 minutes late. Sorry!",
    "Happy birthday! Hope you have an amazing day.",
    "Can you pick up some milk on your way home?",
    "Your doctor's appointment is scheduled for Friday at 3 PM.",
    "Thanks for helping me with the presentation yesterday.",
    "The package arrived this morning. I'll bring it tomorrow.",
    "Please review the attached report before our discussion.",
    "Movie starts at 7 PM. See you at the theater!"
]

print("----------------ham message list---------------------------")
for message_input in ham_message_array:
    text_input = {
        "text": message_input
    }
    response = requests.post(url, json=text_input)
    print(response.json())

ham_message_array = [
    "Congratulations! You've won a FREE iPhone. Click here now: http://bit.ly/freephone",
    "URGENT! Your bank account has been suspended. Verify immediately at http://secure-bank-login.com",
    "You have been selected to receive a $1000 Walmart gift card. Claim now!",
    "WIN CASH NOW! Text WIN to 88888 for your chance to win $10,000.",
    "Exclusive offer! Get approved for a loan instantly with no credit check.",
    "You have 1 unread tax refund notification. Submit your details here immediately.",
    "Limited-time offer! Buy one, get three free. Visit our website today.",
    "Your parcel delivery failed. Confirm your address here: http://delivery-update.example",
    "Act now! Earn $5000 per week working from home. No experience required.",
    "Final notice: Your subscription will expire today. Renew now to avoid service interruption."
]

print("----------------spam message list---------------------------")
for message_input in ham_message_array:
    text_input = {
        "text": message_input
    }
    response = requests.post(url, json=text_input)
    print(response.json())