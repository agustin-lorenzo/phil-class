import gradio as gr
from transformers import pipeline

pipe = pipeline('text-classification', 'model')

def classify(text):
    return pipe(text)[0]['label']

with gr.Blocks() as demo:
    gr.Markdown("# Philosophy Classifier")
    text = gr.Textbox(label="Input")
    classify_btn = gr.Button("Classify Text")
    output = gr.Textbox(label="Predicted Philosophy")
    classify_btn.click(fn=classify, inputs=text, outputs=output, api_name="classify")

demo.title = "Philosophy Classifier"


demo.launch(share=True)