AD Risk App — Early Alzheimer's Risk Assessment (Demo)
=====================================================

Overview
--------
Flask-based demo for early Alzheimer's risk assessment using multi-source speech features (language, paralinguistic, non-linguistic, classic acoustics, eGeMAPS) and a simple graph-based reasoning approach. Optionally blends probability from a local scikit-learn model bundle. ASR is optional via faster-whisper (CTranslate2 weights).

Key Features
------------
- Audio upload (.wav), optional transcript input
- Feature extraction: librosa (MFCC/Chroma/Contrast/Tonnetz), eGeMAPS (openSMILE)
- Graph reasoning to produce an overall risk score and contributions
- Optional scikit-learn model blending if `models/sk_model.joblib` exists
- Optional offline ASR via faster-whisper (local CTranslate2 weights)

Requirements
------------
- Python 3.12
- System packages: `ffmpeg`, `libsndfile1`

Setup
-----
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Run (Development)
-----------------
```bash
source .venv/bin/activate
python app.py
# Health check
curl -sS http://127.0.0.1:5000/health
```

Access in browser:
- Same host: `http://127.0.0.1:5000/`
- Remote access: use `http://<SERVER_IP>:5000/` or create an SSH tunnel:
  ```bash
  ssh -N -L 5000:127.0.0.1:5000 <user>@<SERVER_IP>
  ```

Optional: Offline ASR (faster-whisper)
--------------------------------------
Place CTranslate2 weights of a Whisper model (e.g., `Systran/faster-whisper-small`) under:
```
models/whisper-small/
```
Environment variables (optional):
```bash
export AD_ASR_MODEL_DIR=/root/autodl-tmp/ad-risk-app/models/whisper-small
export AD_ASR_COMPUTE=int8   # int8 / float16 / float32
```
Example download (requires `huggingface_hub` — already installed via `faster-whisper`):
```bash
python - << 'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Systran/faster-whisper-small",
    local_dir="/root/autodl-tmp/ad-risk-app/models/whisper-small",
    allow_patterns=["*.bin","*.json","*.txt","*.model","*tokenizer*"]
)
PY
```

Configuration
-------------
The app reads environment variables in `config.py`:
- `AD_DEMO_MODE` (default 1): demo mode, run without datasets/models
- `AD_USE_TF` (default 0): enable TensorFlow model loading
- `AD_MODEL_PATH`: TF model path (default `models/model.hdf5`)
- `AD_SK_MODEL_PATH`: scikit-learn bundle path (default `models/sk_model.joblib`)
- `AD_ASR_MODEL_DIR`: local path to faster-whisper CTranslate2 weights (default `models/whisper-small`)
- `AD_ASR_COMPUTE`: faster-whisper compute type (`int8`/`float16`/`float32`)

Production Notes
----------------
- Use a production WSGI server (e.g., `gunicorn`) behind a reverse proxy.
- Consider `.env` + `python-dotenv` if you want file-based env management.

Directory Layout (partial)
--------------------------
```
ad-risk-app/
  app.py                # Flask entry
  config.py             # AppConfig + directories and model paths
  features/             # Feature extraction (librosa + openSMILE)
  services/predictor.py # RiskPredictor: graph reasoning + optional SK model
  utils/                # ASR, spectrogram, graph reasoning, vectorizer
  templates/index.html  # UI (Bootstrap)
  static/               # Spectrogram images are saved here
  uploads/              # Uploaded audio files
  models/               # Optional: sk_model.joblib, whisper-small weights, TF model
```

License
-------
MIT. See `LICENSE` for details.

Dataset Usage and Attribution (ADReSSo21 via TalkBank)
-----------------------------------------------------
- The ADReSSo21 Challenge dataset used by this project was obtained via TalkBank. See the TalkBank website for access rules and ground rules: [https://talkbank.org/](https://talkbank.org/).
- The dataset itself is NOT included in this repository. You must obtain access directly from TalkBank/organizers under their Terms of Use/Data Use Agreement (DUA).
- By using ADReSSo21, you agree to comply with TalkBank ground rules and the dataset’s DUA. Typical requirements include (but are not limited to):
  - research-only usage unless otherwise permitted
  - no redistribution of the dataset or any part of it in this repo or releases
  - no attempts to re-identify participants; preserve privacy and ethics
  - follow the required citation/acknowledgement specified by the organizers
- Do not upload raw audio, transcripts, or any redistributable portion of the dataset to this repository.

Suggested citation for the challenge (example):
```
Luz S., Haider F., et al. The ADReSSo Challenge: Detecting cognitive decline using speech only.
Interspeech 2021. (Please adapt to the authoritative bib entry from the organizers.)
```

Acknowledgement statement (example):
```
This work uses the ADReSSo21 Challenge dataset. We thank the organizers and data providers.
All use of the dataset follows the dataset’s terms and privacy/ethics requirements.
Data access and ground rules via TalkBank: https://talkbank.org/
```


