import os
import warnings
import time
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

from config import AppConfig
from services.predictor import RiskPredictor
from utils.spectrogram import save_spectrogram


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = AppConfig.UPLOAD_DIR

predictor = RiskPredictor()


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/health')
def health():
	return jsonify({
		"status": "ok",
		"demo": AppConfig.DEMO_MODE,
		"use_tf": AppConfig.USE_TF,
		"use_sk": os.path.exists(AppConfig.MODEL_SK_PATH),
	})


@app.route('/uploader', methods=['POST'])
def upload_file_action():
	file = request.files.get('file')
	if not file:
		return render_template('index.html', prediction="No audio file selected")
	filename = secure_filename(file.filename)
	audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	file.save(audio_path)

	# Save spectrogram image for frontend display
	img_path = save_spectrogram(audio_path)
	img_rel = '/static/' + os.path.basename(img_path)

	# Run prediction
	text_transcript = request.form.get('transcript', '')
	start = time.time()
	result = predictor.predict(audio_path, text_transcript)
	result["latency_ms"] = int((time.time() - start) * 1000)

	return render_template('index.html', prediction=result.get('label'), image_url=img_rel,
							risk_score=result.get('risk_score'), details=result.get('features'),
							contrib=result.get('contributions'), advice=result.get('advice'), risk_stage=result.get('risk_stage'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
	if 'file' not in request.files:
		return jsonify({"error": "no file"}), 400
	file = request.files['file']
	filename = secure_filename(file.filename)
	audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	file.save(audio_path)

	text_transcript = request.form.get('transcript', '')
	result = predictor.predict(audio_path, text_transcript)
	return jsonify(result)


@app.route('/api/explain', methods=['POST'])
def api_explain():
	if 'file' not in request.files:
		return jsonify({"error": "no file"}), 400
	file = request.files['file']
	filename = secure_filename(file.filename)
	audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	file.save(audio_path)
	text_transcript = request.form.get('transcript', '')
	result = predictor.predict(audio_path, text_transcript)
	return jsonify({
		"risk_score": result.get('risk_score'),
		"label": result.get('label'),
		"contributions": result.get('contributions'),
		"features": result.get('features'),
	})


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)
