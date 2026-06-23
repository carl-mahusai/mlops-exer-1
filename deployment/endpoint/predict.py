from flask import Flask, request, jsonify
import torch
import os
from mlflow.tracking import MlflowClient
import mlflow.artifacts
from pathlib import Path
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from spam_checker.models.spam_classifier import SpamClassifier
from spam_checker.predict_sms import predict_sms

def return_module_args(s):
    # Check for integer first
    try:
        int_variable = int(s)
        return int_variable
    except ValueError:
        pass
    
    # Check for float next
    try:
        float_variable = float(s)
        return float_variable
    except ValueError:
        pass
        
    return s

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

error_message = "model files don't exist"
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

    all_tags = run_info.data.tags

    processing_tag_check = all_tags.get("processing")

    processing = "single"

    if (processing_tag_check):
        processing = processing_tag_check


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

    model_built = False
    if (vocab_path.exists() and model_path.exists()):
        vocab = torch.load(vocab_path)
        if (processing == "single"):
            try:
                

                model = SpamClassifier.load_from_checkpoint(
                    model_path,
                    vocab_size=len(vocab),
                )

                model_built = True
            except Exception as e:
                model_built = False
                error_message = str(e)
                print(error_message)
        else:
            try:
                print("setup multi processing checkpoint load here")

                state_dict = get_fp32_state_dict_from_zero_checkpoint(model_path)

                new_dict = {}

                for key, value in run_info.data.params.items():
                    print(f"Key: {key} | Value: {value}")
                    new_dict_value = return_module_args(value)
                    new_dict[key] = new_dict_value

                print(new_dict)
                try:
                    model = SpamClassifier(**new_dict)
                    model.load_state_dict(state_dict)
                    model_built = True
                except Exception as e:
                    model_built = False
                    error_message = str(e)
                    print("inner except - initial build from dict failed - ", error_message, sep="\n")
                    print("attempting backup model build")
                    try:
                        model = SpamClassifier(
                            vocab_size=len(vocab)
                        )
                        model.load_state_dict(state_dict)
                        model_built = True
                    except Exception as e:
                        model_built = False
                        error_message = str(e)
                        print("inner except - final build attempt failed - ", error_message, sep="\n")
            except Exception as e:
                model_built = False
                error_message = str(e)
                print("outer except - ", error_message, sep="\n")

app = Flask('spam-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    text = request.get_json().get("text")

    if (model_built):
        result = predict_sms(
            text,
            model,
            vocab,
        )

        result["response"] = "ok"
    else:
        result = {
            "response": "nok",
            "error_message": error_message
        }

    # print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
