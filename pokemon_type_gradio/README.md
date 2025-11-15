Gradio app for Pokemon Type Classifier

Files:
- `model_wrapper_gradio.py`: loads the model checkpoint and runs preprocessing + inference
- `app_gradio.py`: Gradio Blocks UI to upload image and display predictions
- `requirements.txt`: minimal dependencies

Quick start

1. Create a virtualenv and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Copy your checkpoint into `checkpoints/best_model.ckpt` (or set `MODEL_PATH` env var)

```bash
mkdir -p checkpoints
cp ../checkpoints/your_best_model.ckpt checkpoints/best_model.ckpt
```

3. Run the app

```bash
export MODEL_PATH=checkpoints/best_model.ckpt
export DEVICE=cpu  # or 'cuda' if GPU available
python app_gradio.py
```

Open the URL printed by Gradio (default http://0.0.0.0:7860)

Notes
- The wrapper attempts to load both Lightning-style checkpoints (which store `state_dict`) and plain state_dicts. If loading fails, adjust `model_wrapper_gradio.py` to match whichever keys your checkpoint uses.
- The model and transforms mirror the notebook training setup (resize to 128 and normalization values).
- Threshold default is 0.5; you can adjust it in the UI.

If you'd like, I can:
- Add a small demo image set with examples
- Add Dockerfile and deployable container
- Add a server-side batch/predict endpoint (non-interactive)
