# LightURLNet Streamlit App

Local Streamlit app for testing URLs with the included `lighturlnet.onnx` model.

## Files

- `app.py` - Streamlit UI for single URL and batch scanning.
- `inference.py` - ONNX Runtime inference helper.
- `lighturlnet.onnx` - model artifact used by the app.
- `vocab.json` and `config.json` - tokenizer vocabulary and model settings.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal.

## Class Order

The app assumes ONNX output class index `1` means danger. If your training labels used the
opposite order, run with:

```powershell
$env:LIGHTURLNET_DANGER_CLASS_INDEX = "0"
streamlit run app.py
```

## Git

The ONNX model is intentionally tracked because it is small enough for a normal Git push.
