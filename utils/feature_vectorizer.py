from typing import Dict, Any, List, Optional
import numpy as np



_FEATURES: List[str] = [
	# language
	"language.token_count",
	"language.avg_token_len",
	# paralinguistic
	"paralinguistic.rms_energy",
	"paralinguistic.tempo",
	"paralinguistic.f0_mean",
	# nonlinguistic
	"nonlinguistic.spectral_centroid",
	"nonlinguistic.spectral_bandwidth",
	"nonlinguistic.zero_crossing_rate",
]

# Extended acoustic features (aligned with extract_acoustic_features)
_FEATURES += [f"acoustic.mfcc_mean_{i}" for i in range(13)]
_FEATURES += [f"acoustic.mfcc_std_{i}" for i in range(13)]
_FEATURES += [f"acoustic.chroma_mean_{i}" for i in range(12)]
_FEATURES += [f"acoustic.contrast_mean_{i}" for i in range(7)]
_FEATURES += [f"acoustic.tonnetz_mean_{i}" for i in range(6)]


def feature_names() -> List[str]:
	return list(_FEATURES)


def feats_to_vector(features: Dict[str, Any], wavlm_vec: Optional[np.ndarray] = None) -> np.ndarray:
	def get(path: str) -> float:
		section, key = path.split(".", 1)
		sec = features.get(section, {})
		try:
			return float(sec.get(key, 0.0))
		except Exception:
			return 0.0

	values = [get(name) for name in _FEATURES]
	base = np.asarray(values, dtype=np.float32)
	# 追加 eGeMAPS（若存在则按 key 排序，保证顺序一致）
	egemaps = features.get('egemaps', {}) or {}
	if egemaps:
		keys = sorted(list(egemaps.keys()))
		egemaps_vals = np.asarray([float(egemaps.get(k, 0.0)) for k in keys], dtype=np.float32)
		base = np.concatenate([base, egemaps_vals], axis=0)
	if wavlm_vec is not None and wavlm_vec.size > 0:
		return np.concatenate([base, wavlm_vec.astype(np.float32)], axis=0)
	return base


