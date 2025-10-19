import os
import numpy as np
from typing import Dict, Any
from config import AppConfig
from features.extractors import extract_all
from utils.graph_reasoning import graph_risk_and_contributions
from joblib import load

try:
	if AppConfig.USE_TF:
		from tensorflow.keras.models import load_model as tf_load_model
	else:
		tf_load_model = None
except Exception:
	tf_load_model = None


class RiskPredictor:
	def __init__(self) -> None:
		self.demo_mode = AppConfig.DEMO_MODE
		self.tf_model = None
		self.sk_model = None
		# Try load TensorFlow model when enabled
		if AppConfig.USE_TF and tf_load_model is not None and os.path.exists(AppConfig.MODEL_PATH):
			self.tf_model = tf_load_model(AppConfig.MODEL_PATH, compile=False)
		# Try load scikit-learn model bundle if available
		if os.path.exists(AppConfig.MODEL_SK_PATH):
			try:
				bundle = load(AppConfig.MODEL_SK_PATH)
				self.sk_model = bundle.get('model', None)
				print(f"[RiskPredictor] 已加载sk模型: {AppConfig.MODEL_SK_PATH}", flush=True)
			except Exception:
				self.sk_model = None

	def _blend_scores(self, graph_score: float, sk_prob: float | None) -> float:
		if sk_prob is None:
			return graph_score
		return float(np.clip(0.6 * graph_score + 0.4 * sk_prob, 0.0, 1.0))

	def _stage_and_advice(self, score: float) -> tuple[str, str]:
		# Three-stage risk stratification. Calibrate as needed: <0.33 low; 0.33-0.66 medium; ≥0.66 high
		if score < 0.33:
			stage = "低风险"
			advice = (
				"Maintain a healthy lifestyle (sleep routine, exercise, balanced diet), "
				"monitor memory and language changes; recheck every 1-3 months."
			)
		elif score < 0.66:
			stage = "中度风险"
			advice = (
				"Perform brief cognitive screening (e.g., MoCA/MMSE) soon; address sleep/emotion; "
				"retest in 2-4 weeks or combine with caregiver observations."
			)
		else:
			stage = "高风险"
			advice = (
				"Visit memory clinic/neurology for professional assessment (neuropsychological tests) ASAP; "
				"use imaging/blood/EEG when needed; this system is auxiliary only."
			)
		return stage, advice

	def predict(self, audio_path: str, text_transcript: str = "") -> Dict[str, Any]:
		feats = extract_all(audio_path, text_transcript)
		# Graph-based reasoning (cognitive-paralinguistic-physiological + cascade edges)
		overall_graph, contrib = graph_risk_and_contributions(feats)
		# scikit-learn probability
		sk_prob = None
		try:
			if self.sk_model is not None:
				from utils.feature_vectorizer import feats_to_vector
				# Explicitly disable WavLM (avoid network); use local features only
				vec = feats_to_vector(feats, wavlm_vec=None).reshape(1, -1)
				proba = self.sk_model.predict_proba(vec)
				sk_prob = float(proba[0, 1])
		except Exception:
			sk_prob = None

		score = self._blend_scores(overall_graph, sk_prob)
		stage, advice = self._stage_and_advice(score)
		label = f"{stage}（Alzheimer-related）"
		return {
			"risk_score": round(float(score), 3),
			"label": label,
			"features": feats,
			"contributions": contrib,
			"sk_prob": None if sk_prob is None else round(float(sk_prob), 3),
			"risk_stage": stage,
			"advice": advice,
		}
