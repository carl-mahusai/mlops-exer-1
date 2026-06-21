import gradio as gr
import requests

API_URL = "http://localhost:9696/predict"

def classify_text(text):
    payload = {
        "text": text
    }

    response = requests.post(API_URL, json=payload)

    # Raise exception if request failed
    response.raise_for_status()

    result = response.json()

    return result["prediction"]

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Message"),
    outputs=gr.Textbox(label="Prediction"),
    title="Spam Classifier"
)

demo.launch()