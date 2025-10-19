import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

from config import AppConfig


def save_spectrogram(audio_path: str) -> str:
	filename = os.path.basename(audio_path)
	stem = os.path.splitext(filename)[0]
	output_path = os.path.join(AppConfig.STATIC_DIR, f"{stem}.jpg")
	y, sr = librosa.load(audio_path, sr=None)
	# Use mel-spectrogram for better perceptual alignment with human hearing
	S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
	S_db = librosa.power_to_db(S, ref=librosa.power_to_db)
	plt.figure(figsize=(10, 3))
	librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap='magma')
	plt.axis('off')
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()
	return output_path
