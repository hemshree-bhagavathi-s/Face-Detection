"""
Face Detection System - Python Flask Backend API Example
Endpoint: POST /predict
"""

import base64
import io
import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for browser access

# Try loading the trained CNN model (face_detector.h5)
model = None
MODEL_PATH = 'face_detector.h5'

if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ CNN Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"⚠️ Could not load TensorFlow model ({e}). Fallback mode active.")
else:
    print(f"ℹ️ '{MODEL_PATH}' not found in current directory. Running in simulation mode.")


def preprocess_image(image_bytes, target_size=(224, 224)):
    """Convert raw image bytes to normalized numpy array for CNN model."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image field in request body'}), 400

        base64_str = data['image']
        
        # Strip header if data URI scheme is included (e.g. data:image/jpeg;base64,...)
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        image_bytes = base64.b64decode(base64_str)

        if model is not None:
            # Perform prediction using loaded CNN model
            input_tensor = preprocess_image(image_bytes)
            raw_prediction = model.predict(input_tensor, verbose=0)[0]
            
            # Assuming output is probability of face detection
            score = float(raw_prediction[0] if hasattr(raw_prediction, '__len__') else raw_prediction)
            face_detected = score >= 0.5
            confidence = round(score * 100, 2) if face_detected else round((1 - score) * 100, 2)

        else:
            # Fallback simulated prediction if TensorFlow is not installed
            # Checks image entropy / size as lightweight heuristic for simulation
            img = Image.open(io.BytesIO(image_bytes))
            stat_std = np.std(np.array(img))
            face_detected = bool(stat_std > 20)
            confidence = round(85.0 + (stat_std % 14.0), 1) if face_detected else round(15.0 + (stat_std % 20.0), 1)

        return jsonify({
            'face_detected': face_detected,
            'confidence': confidence
        }), 200

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'service': 'Face Detection CNN Flask API',
        'model_loaded': model is not None
    }), 200


if __name__ == '__main__':
    print("🚀 Starting Flask Face Detection API on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
