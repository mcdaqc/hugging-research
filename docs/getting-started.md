# Getting started

## Requirements
- Python 3.10+
- Internet connection

## Install
```bash
git clone https://github.com/mcdaqc/hugging-research
cd hugging-research
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
pip install -r requirements.txt
```

## Configure
Create `.env` (token is only for the inference model; tools are anonymous):
```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MODEL_ID=Qwen/Qwen3-Coder-480B-A35B-Instruct
```

## Run
```bash
python app.py
# open http://localhost:7860
```

## Notes for Spaces
- Set `HF_TOKEN` as a Space Secret if you want to choose a different inference model via `MODEL_ID`.
- Tools never use tokens; private/gated items will be marked as not accessible.
