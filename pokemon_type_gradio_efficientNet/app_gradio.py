import os
import gradio as gr
import pandas as pd
from model_wrapper_gradio import GradioPokemonPredictor
from PIL import Image
import tempfile

# Set Gradio temp directory to avoid permission issues
TEMP_DIR = os.path.join(os.getcwd(), 'gradio_temp')
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = TEMP_DIR

MODEL_PATH = os.environ.get('MODEL_PATH', 'checkpoints/best_model.ckpt')
DEVICE = os.environ.get('DEVICE', 'cpu')

# Initialize predictor
print(f"Loading model from: {MODEL_PATH}")
print(f"Using device: {DEVICE}")
predictor = GradioPokemonPredictor(MODEL_PATH, device=DEVICE)
print("Model loaded successfully!")

def predict_image(image: Image.Image, threshold: float = 0.5, top_k: int = 3):
    """
    Predict Pokemon types from an image
    
    Args:
        image: PIL Image from Gradio
        threshold: Minimum probability threshold for predictions
        top_k: Number of top predictions to return
    
    Returns:
        Tuple of (bar_plot_data, above_threshold_dict, topk_dict)
    """
    if image is None:
        return None, {"error": "No image provided"}, {"error": "No image provided"}
    
    try:
        # Get predictions
        above = predictor.predict_pil(image, threshold=threshold)
        topk = predictor.predict_topk(image, k=top_k)
        total = predictor.predict_full(image)
        
        # Build DataFrame for BarPlot
        totals_for_plot = pd.DataFrame([
            {'type': t, 'confidence': float(p)} for t, p in total
        ])
        
        # Format other outputs as JSON-friendly dicts
        out_above = {
            'count': len(above),
            'predictions': [{'type': t, 'probability': f"{float(p):.4f}"} for t, p in above]
        }
        out_topk = {
            'predictions': [{'type': t, 'probability': f"{float(p):.4f}"} for t, p in topk]
        }
        
        return totals_for_plot, out_above, out_topk
    
    except Exception as e:
        error_msg = {"error": str(e)}
        return None, error_msg, error_msg


# Create Gradio interface
with gr.Blocks(title="Pokemon Type Classifier") as demo:
    gr.Markdown("""
    # 🎮 Pokemon Type Classifier
    ### EfficientNetB3 Multi-label Classification
    
    Upload an image of a Pokemon to predict its type(s). This model can predict multiple types for dual-type Pokemon.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type='pil', label='Upload Pokemon Image')
            
            with gr.Row():
                threshold = gr.Slider(
                    minimum=0.1, 
                    maximum=0.9, 
                    step=0.05, 
                    value=0.5, 
                    label='Prediction Threshold',
                    info="Minimum confidence to include a type"
                )
                topk = gr.Slider(
                    minimum=1, 
                    maximum=6, 
                    step=1, 
                    value=3, 
                    label='Top-K Predictions',
                    info="Number of top types to show"
                )
            
            btn = gr.Button('🔍 Predict Type', variant='primary', size='lg')
        
        with gr.Column(scale=1):
            totals = gr.BarPlot(
                x="type", 
                y="confidence", 
                title="All Type Probabilities",
                y_title="Confidence",
                x_title="Pokemon Type",
                height=400
            )
    
    with gr.Row():
        with gr.Column():
            out_above = gr.JSON(label='Predictions Above Threshold')
        with gr.Column():
            out_topk = gr.JSON(label='Top-K Predictions')
    
    gr.Markdown("""
    ### About This Model
    - **Architecture**: EfficientNetB3 with custom classifier head
    - **Task**: Multi-label classification (Pokemon can have 1-2 types)
    - **Classes**: 18 Pokemon types
    """)
    
    # Connect button to prediction function
    btn.click(
        fn=predict_image, 
        inputs=[img_in, threshold, topk], 
        outputs=[totals, out_above, out_topk]
    )
    
    # Optional: Add example images if you have them
    # gr.Examples(
    #     examples=[
    #         ["examples/pikachu.jpg"],
    #         ["examples/charizard.jpg"],
    #     ],
    #     inputs=img_in
    # )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7861))
    demo.launch(server_name='0.0.0.0', server_port=port)