import argparse
import requests
import gradio as gr

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--api-url",
    type=str,
    required=True,
    help="URL of the backend API endpoint"
)

args = parser.parse_args()

API_URL = args.api_url


def predict(text):
    response = requests.post(
        API_URL,
        json={"text": text}
    )
    response.raise_for_status()

    return response.json()["prediction"]


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Prediction"),
    title="Spam Classifier"
)

if __name__ == "__main__":
    demo.launch()