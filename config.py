import os


class AppConfig:
	# Demo mode: runnable without datasets and models
	DEMO_MODE: bool = os.environ.get("AD_DEMO_MODE", "1") == "1"

	# Directories
	BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
	UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
	STATIC_DIR: str = os.path.join(BASE_DIR, "static")
	MODELS_DIR: str = os.path.join(BASE_DIR, "models")

	# Models
	MODEL_PATH: str = os.environ.get("AD_MODEL_PATH", os.path.join(MODELS_DIR, "model.hdf5"))
	MODEL_SK_PATH: str = os.environ.get("AD_SK_MODEL_PATH", os.path.join(MODELS_DIR, "sk_model.joblib"))
	USE_TF: bool = os.environ.get("AD_USE_TF", "0") == "1"

	# Labels
	CATEGORIES = {0: "无阿尔茨海默高风险", 1: "存在阿尔茨海默高风险"}


os.makedirs(AppConfig.UPLOAD_DIR, exist_ok=True)
os.makedirs(AppConfig.STATIC_DIR, exist_ok=True)
os.makedirs(AppConfig.MODELS_DIR, exist_ok=True)
