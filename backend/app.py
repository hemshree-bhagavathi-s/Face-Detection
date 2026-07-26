"""
==============================================================================
Face Detection System - Python Flask Backend API
==============================================================================
"""

import os
import base64
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], "GPU")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# Flask App
app = Flask(__name__)
CORS(app)

# ============================================================================
# Load Haar Cascade
# ============================================================================

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade.")

# ============================================================================
# Load TensorFlow Lite Model
# ============================================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "face_detector.tflite"
)

TARGET_SIZE = (128, 128)

try:
    cnn_model = tf.lite.Interpreter(model_path=MODEL_PATH)
    cnn_model.allocate_tensors()

    input_details = cnn_model.get_input_details()
    output_details = cnn_model.get_output_details()

    logging.info("TensorFlow Lite model loaded successfully.")

except Exception as e:
    cnn_model = None
    logging.error(e)

# ============================================================================
# Helper Functions
# ============================================================================

def decode_base64_image(base64_string):

    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    img_bytes = base64.b64decode(base64_string)

    img_np = np.frombuffer(img_bytes, np.uint8)

    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")

    return image


def preprocess_face_crop(face):

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = cv2.resize(face, TARGET_SIZE)

    face = face.astype(np.float32) / 255.0

    face = np.expand_dims(face, axis=0)

    return face

# ============================================================================
# API Routes
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "Face Detection System CNN API",
        "model_loaded": cnn_model is not None,
        "cascade_loaded": not face_cascade.empty(),
        "input_resolution": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if cnn_model is None:
        return jsonify({
            "face_detected": False,
            "confidence": 0,
            "error": "Model not loaded"
        }), 500

    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "face_detected": False,
                "confidence": 0,
                "error": "No image received"
            }), 400

        image = decode_base64_image(data["image"])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )

        # No face found
        if len(faces) == 0:
            return jsonify({
                "face_detected": False,
                "confidence": 0
            }), 200

        # Largest face
        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        pad = 10

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            return jsonify({
                "face_detected": False,
                "confidence": 0
            }), 200

        input_tensor = preprocess_face_crop(face)

        cnn_model.set_tensor(
            input_details[0]["index"],
            input_tensor.astype(np.float32)
        )

        cnn_model.invoke()

        prediction = cnn_model.get_tensor(
            output_details[0]["index"]
        )

        score = float(prediction[0][0])

        print("Prediction Score:", score)

        THRESHOLD = 0.60

        face_detected = score < THRESHOLD

        if face_detected:
            confidence = round((1 - score) * 100, 1)
        else:
            confidence = round(score * 100, 1)

        return jsonify({
            "face_detected": face_detected,
            "confidence": confidence
        }), 200

    except Exception as e:

        logging.exception(e)

        return jsonify({
            "face_detected": False,
            "confidence": 0,
            "error": str(e)
        }), 500

    # ============================================================================
# Custom Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Route not found",
        "message": "Valid endpoints: GET / and POST /predict"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error"
    }), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )