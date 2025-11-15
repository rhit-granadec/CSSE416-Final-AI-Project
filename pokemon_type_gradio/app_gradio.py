import os
import gradio as gr
import pandas as pd
from model_wrapper_gradio import GradioPokemonPredictor
from PIL import Image

MODEL_PATH = os.environ.get('MODEL_PATH', 'checkpoints/best_model.ckpt')
DEVICE = os.environ.get('DEVICE', 'cpu')

# Initialize predictor
predictor = GradioPokemonPredictor(MODEL_PATH, device=DEVICE)


def predict_image(image: Image.Image, threshold: float = 0.5, top_k: int = 3):
    # image is a PIL image provided by Gradio
    above = predictor.predict_pil(image, threshold=threshold)
    topk = predictor.predict_topk(image, k=top_k)
    total = predictor.predict_full(image)

    # Build values in the format expected by gr.BarPlot
    totals_for_plot = pd.DataFrame([
        {'type': t, 'confidence': float(p)} for t, p in total
    ])

    # Keep the other outputs as JSON-friendly dicts
    out_above = [{'type': t, 'probability': float(p)} for t, p in above]
    out_topk = [{'type': t, 'probability': float(p)} for t, p in topk]

    # Return values
    return [totals_for_plot, {'out_above': out_above}, {'out_topk': out_topk}]


with gr.Blocks() as demo:
    gr.Markdown("# Pokemon Type Classifier (Gradio)")
    with gr.Row():
        img_in = gr.Image(type='pil', label='Upload Pokemon image')
        with gr.Column():
            threshold = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.5, label='Prediction threshold')
            topk = gr.Slider(minimum=1, maximum=6, step=1, value=3, label='Top-k')
            btn = gr.Button('Predict')
    totals = gr.BarPlot(x="type", y="confidence", label='Ordered probability list', sort='-y')
    out_above = gr.JSON(label='Predictions above threshold')
    out_topk = gr.JSON(label='Top-k predictions')

    btn.click(fn=predict_image, inputs=[img_in, threshold, topk], outputs=[totals, out_above, out_topk])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    demo.launch(server_name='0.0.0.0', server_port=port)