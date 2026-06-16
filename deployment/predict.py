from flask import Flask, request, jsonify
import torch
import os
from mlflow.tracking import MlflowClient
import mlflow.artifacts

from spam_checker.models.spam_classifier import SpamClassifier
from spam_checker.predict_sms import predict_sms

# add section here which would retrieve checkpoint and vocab file to set up model

RUN_ID = os.getenv('RUN_ID')

print(RUN_ID)

TRACKING_URI = os.getenv('TRACKING_URI')

print(TRACKING_URI)

if (len(RUN_ID) > 0 and len(TRACKING_URI) > 0):
    # client = MlflowClient()
    # local_path = client.download_artifacts(run_id=RUN_ID, path="train.csv", dst_path=".")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # run_id = "YOUR_RUN_ID_HERE"
    artifact_path = "evaluation"  # Downloads everything in the run root directory
    local_dir = ""

    # Download using the specified client instance
    mlflow.artifacts.download_artifacts(
        run_id=RUN_ID,
        artifact_path=artifact_path,
        dst_path=local_dir,
        tracking_uri=client.tracking_uri
    )


# vocab = torch.load("vocab.pt")

# model = SpamClassifier.load_from_checkpoint(
#     "sms_spam.ckpt",
#     vocab_size=len(vocab),
# )



app = Flask('spam-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    text = request.get_json()

    # result = predict_sms(
    #     text,
    #     model,
    #     vocab,
    # )

    result = {
        "text": "test"
    }

    # print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
