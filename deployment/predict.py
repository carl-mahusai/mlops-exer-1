from flask import Flask, request, jsonify
import torch

from spam_checker.models.spam_classifier import SpamClassifier

# add section here which would retrieve checkpoint and vocab file to set up model

vocab = torch.load("../vocab.pt")

model = SpamClassifier.load_from_checkpoint(
    "../sms_spam.ckpt",
    vocab_size=len(vocab),
)

app = Flask('spam-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    text = request.get_json()

    print("text")

    # return jsonify({
    #     'result': "test"
    # })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
