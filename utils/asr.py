import os
from typing import Optional


def transcribe_or_none(audio_path: str) -> Optional[str]:
    """Offline transcription using faster-whisper with local CTranslate2 weights.

    Directory priority:
    1) Environment variable AD_ASR_MODEL_DIR
    2) Default path ./models/whisper-small
    If neither is available, return None so upper layers fall back to proxy language features.
    """
    try:
        from faster_whisper import WhisperModel
        # Compute project root (parent of utils/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_local = os.path.join(project_root, 'models', 'whisper-small')
        model_dir = (os.environ.get('AD_ASR_MODEL_DIR', '').strip() or default_local)
        compute_type = os.environ.get('AD_ASR_COMPUTE', 'int8').strip()
        if not os.path.isdir(model_dir):
            return None
        model = WhisperModel(model_dir, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=1, vad_filter=True)
        texts = [getattr(seg, 'text', '').strip() for seg in segments if getattr(seg, 'text', '').strip()]
        return ' '.join(texts) if texts else None
    except Exception:
        return None



