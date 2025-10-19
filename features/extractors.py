import numpy as np
import librosa
from typing import Dict, Any, Tuple
import warnings


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
	x, sr = librosa.load(audio_path, sr=target_sr)
	return x, sr


def extract_language_features(text_transcript: str = "") -> Dict[str, Any]:
	# Text-based language statistics
	if text_transcript:
		tokens = text_transcript.strip().split()
		avg_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
		return {"token_count": len(tokens), "avg_token_len": avg_len}
	return {"token_count": 0, "avg_token_len": 0.0}


def extract_language_proxy_from_audio(signal: np.ndarray, sr: int) -> Dict[str, Any]:
    """When ASR is unavailable, estimate a proxy of language fluency based on
    short-time energy changes and pauses. Returns token_count and avg_token_len
    proxies to ensure the language section is non-zero.
    """
    # Simple strategy: threshold short-frame energy to count segments and length
	frame = max(sr // 20, 1)
	energy = np.array([np.mean(np.abs(signal[i:i+frame])) for i in range(0, len(signal), frame)])
	thr = float(np.median(energy) * 1.2)
	flags = (energy > thr).astype(np.int32)
	# 统计连续“1”的段落数与长度
	segments = []
	count = 0
	for v in flags:
		if v == 1:
			count += 1
		else:
			if count > 0:
				segments.append(count)
				count = 0
	if count > 0:
		segments.append(count)
	seg_count = len(segments)
	avg_seg = float(np.mean(segments)) if segments else 0.0
	return {"token_count": seg_count, "avg_token_len": avg_seg}


def extract_paralinguistic_features(signal: np.ndarray, sr: int) -> Dict[str, Any]:
	energy = float(np.mean(librosa.feature.rms(y=signal)))
	tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
	f0 = librosa.yin(signal, fmin=50, fmax=500, sr=sr)
	f0_mean = float(np.nanmean(f0)) if np.any(np.isfinite(f0)) else 0.0
	return {
		"rms_energy": energy,
		"tempo": float(tempo),
		"f0_mean": f0_mean,
	}


def extract_nonlinguistic_features(signal: np.ndarray, sr: int) -> Dict[str, Any]:
	spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)))
	spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)))
	zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)))
	return {
		"spectral_centroid": spec_centroid,
		"spectral_bandwidth": spec_bw,
		"zero_crossing_rate": zcr,
	}



def extract_acoustic_features(signal: np.ndarray, sr: int) -> Dict[str, Any]:
	# Rich and robust acoustic statistical features
	features: Dict[str, Any] = {}
	# Mean and std of 13 MFCC coefficients
	mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
	features.update({f"mfcc_mean_{i}": float(np.mean(mfcc[i])) for i in range(mfcc.shape[0])})
	features.update({f"mfcc_std_{i}": float(np.std(mfcc[i])) for i in range(mfcc.shape[0])})
	# Mean of 12-d chroma
	chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
	features.update({f"chroma_mean_{i}": float(np.mean(chroma[i])) for i in range(chroma.shape[0])})
	# Mean spectral contrast (typically 7 dims)
	contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
	features.update({f"contrast_mean_{i}": float(np.mean(contrast[i])) for i in range(contrast.shape[0])})
	# Mean tonnetz 6 dims (tonal relation)
	harm = librosa.effects.harmonic(signal)
	try:
		tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)
		features.update({f"tonnetz_mean_{i}": float(np.mean(tonnetz[i])) for i in range(tonnetz.shape[0])})
	except Exception:
		pass
	return features


def extract_opensmile_egemaps(signal: np.ndarray, sr: int) -> Dict[str, Any]:
	"""Reproducible industry standard eGeMAPS via openSMILE.
	If openSMILE environment fails, return empty dict to keep the main flow working.
	"""
	try:
		import opensmile
		smile = opensmile.Smile(
			feature_set=opensmile.FeatureSet.eGeMAPSv02,
			feature_level=opensmile.FeatureLevel.Functionals,
		)
		# openSMILE expects (n,) or (n,1)
		import pandas as pd
		arr = signal.astype(np.float32)
		# Convert numpy to pandas Series for processing (sample rate via arg)
		feat_df = smile.process_signal(arr, sr)
		vals = feat_df.iloc[0].to_dict()
		# Cast numpy scalars to floats
		return {str(k): float(v) for k, v in vals.items()}
	except Exception as e:
		warnings.warn(f'openSMILE extraction failed: {e}')
		return {}


def extract_all(audio_path: str, text_transcript: str = "") -> Dict[str, Any]:
	signal, sr = load_audio(audio_path)
	# 1) Automatic ASR
	from utils.asr import transcribe_or_none
	asr_text = text_transcript.strip() if text_transcript else (transcribe_or_none(audio_path) or "")
	lang = extract_language_features(asr_text)
	# If still empty, fill with audio-derived proxy features
	if lang.get("token_count", 0) == 0:
		lang = extract_language_proxy_from_audio(signal, sr)
	# 2) Other dimensions
	para = extract_paralinguistic_features(signal, sr)
	nonling = extract_nonlinguistic_features(signal, sr)
	acoustic = extract_acoustic_features(signal, sr)
	egemaps = extract_opensmile_egemaps(signal, sr)
	return {"language": lang, "paralinguistic": para, "nonlinguistic": nonling, "acoustic": acoustic, "egemaps": egemaps, "sr": sr}
