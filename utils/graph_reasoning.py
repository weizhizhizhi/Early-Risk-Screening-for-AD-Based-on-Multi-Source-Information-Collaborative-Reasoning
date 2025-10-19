from typing import Dict, Any, Tuple
import numpy as np


def _normalize(value: float, vmin: float, vmax: float) -> float:
	if vmax <= vmin:
		return 0.0
	x = (value - vmin) / (vmax - vmin)
	return float(np.clip(x, 0.0, 1.0))


def _cognitive_score(lang: Dict[str, Any]) -> float:
	# Fewer tokens and shorter average word length → cognitive decline tendency
	tokens = float(lang.get("token_count", 0.0))
	avg_len = float(lang.get("avg_token_len", 0.0))
	# Typical ranges: tokens [0, 120], avg_len [0, 6]
	low_tokens = 1.0 - _normalize(tokens, 20.0, 120.0)
	short_words = 1.0 - _normalize(avg_len, 2.5, 6.0)
	return float(np.clip(0.6 * low_tokens + 0.4 * short_words, 0.0, 1.0))


def _paralinguistic_score(para: Dict[str, Any]) -> float:
	# Too slow/fast tempo, low pitch → risk tendency
	tempo = float(para.get("tempo", 0.0))  # BPM
	f0_mean = float(para.get("f0_mean", 0.0))
	# Typical ranges: tempo [40, 180], f0_mean [80, 300]
	# Both too slow and too fast as abnormal; distance from center 100
	tempo_center = 100.0
	tempo_dev = abs(tempo - tempo_center)
	tempo_risk = _normalize(tempo_dev, 0.0, 60.0)
	low_f0 = 1.0 - _normalize(f0_mean, 120.0, 260.0)
	return float(np.clip(0.7 * tempo_risk + 0.3 * low_f0, 0.0, 1.0))


def _physiological_score(nonling: Dict[str, Any]) -> float:
	# Low spectral centroid, abnormal ZCR → risk tendency
	centroid = float(nonling.get("spectral_centroid", 0.0))
	zcr = float(nonling.get("zero_crossing_rate", 0.0))
	# Typical ranges: centroid [800, 4500] Hz, zcr [0.01, 0.2]
	low_centroid = 1.0 - _normalize(centroid, 1500.0, 3500.0)
	# Both too low and too high ZCR are not ideal; center at 0.07
	zcr_center = 0.07
	zcr_dev = abs(zcr - zcr_center)
	zcr_risk = _normalize(zcr_dev, 0.0, 0.08)
	return float(np.clip(0.6 * low_centroid + 0.4 * zcr_risk, 0.0, 1.0))


def graph_risk_and_contributions(features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
	"""Compute overall risk and contributions based on three nodes
	(cognitive, paralinguistic, physiological) and cascade interactions.

	Returns:
	- overall_risk: 0-1
	- contributions: contribution share of nodes and interactions
	"""
	lang = features.get("language", {})
	para = features.get("paralinguistic", {})
	nonling = features.get("nonlinguistic", {})

	cog = _cognitive_score(lang)
	par = _paralinguistic_score(para)
	phy = _physiological_score(nonling)

	# Node weights (clinical prior: paralinguistic/cognitive slightly higher)
	w_cog, w_par, w_phy = 0.35, 0.4, 0.25
	node_sum = w_cog * cog + w_par * par + w_phy * phy

	# Cascade effects: cognitive↔paralinguistic, paralinguistic↔physiological
	inter_cog_par = cog * par
	inter_par_phy = par * phy
	w_inter = 0.2  # Global interaction weight
	inter_sum = w_inter * 0.5 * (inter_cog_par + inter_par_phy)

	overall = float(np.clip(node_sum + inter_sum, 0.0, 1.0))

	# Split contributions by weighted terms and normalize
	parts = {
		"cognitive": w_cog * cog,
		"paralinguistic": w_par * par,
		"physiological": w_phy * phy,
		"cog↔par": w_inter * 0.5 * inter_cog_par,
		"par↔phy": w_inter * 0.5 * inter_par_phy,
	}
	total = sum(parts.values()) or 1.0
	contrib = {k: float(np.clip(v / total, 0.0, 1.0)) for k, v in parts.items()}

	return overall, contrib


