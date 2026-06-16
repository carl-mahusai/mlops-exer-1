from flask import Flask, request, jsonify
import torch
import os
from mlflow.tracking import MlflowClient
import mlflow.artifacts
from pathlib import Path

from spam_checker.models.spam_classifier import SpamClassifier
from spam_checker.predict_sms import predict_sms

def list_all_artifacts_recursive(client, run_id):
    # client = MlflowClient()
    all_artifacts = []

    def _walk(path=""):
        # Fetch artifacts at the current directory level
        artifacts_level = client.list_artifacts(run_id, path=path)
        for artifact in artifacts_level:
            if artifact.is_dir:
                # If it's a directory, dive deeper
                _walk(artifact.path)
            else:
                # If it's a file, record its path
                all_artifacts.append(artifact.path)

    _walk()
    return all_artifacts

# add section here which would retrieve checkpoint and vocab file to set up model

RUN_ID = os.getenv('RUN_ID')

print(RUN_ID)

TRACKING_URI = os.getenv('TRACKING_URI')

print(TRACKING_URI)

script_dir = Path(__file__).resolve().parent
artifact_path = "evaluation"  # Downloads everything in the run root directory
local_dir = script_dir / "artifact_download"

if (len(RUN_ID) > 0 and len(TRACKING_URI) > 0):
    # client = MlflowClient()
    # local_path = client.download_artifacts(run_id=RUN_ID, path="train.csv", dst_path=".")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    run_info = client.get_run(RUN_ID)

    print("Metadata:", run_info.info)
    print("Parameters:", run_info.data.params)
    print("Metrics:", run_info.data.metrics)
    print("Tags:", run_info.data.tags)
    print(run_info.info.artifact_uri)



    # Download using the specified client instance
    mlflow.artifacts.download_artifacts(
        run_id=RUN_ID,
        artifact_path=artifact_path,
        dst_path=local_dir,
        tracking_uri=client.tracking_uri
    )

    # artifacts_list = list_all_artifacts_recursive(
    #     client=client,
    #     run_id=RUN_ID
    # )

    # print(f"Found {len(artifacts_list)} total artifacts:")
    # for file_path in artifacts_list:
    #     print(f" - {file_path}")


    vocab_path = local_dir / artifact_path / "vocab.pt"
    model_path = local_dir / artifact_path / "sms_spam.ckpt"

    vocab = torch.load(vocab_path)

    model = SpamClassifier.load_from_checkpoint(
        model_path,
        vocab_size=len(vocab),
    )

app = Flask('spam-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    text = request.get_json()

    result = predict_sms(
        text,
        model,
        vocab,
    )

    # result = {
    #     "text": "test"
    # }

    # print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
